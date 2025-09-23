from typing import Dict, Any
from ..models import MetricResult, ModelContext
from ..utils import measure_time
from ..git_inspect import GitInspector
from .base import BaseMetric


class BusFactorMetric(BaseMetric):
    #evaluating contributor diversity and project sustainability
    
    @property
    def name(self) -> str:
        return "bus_factor"
    
    async def compute(self, context: ModelContext, config: Dict[str, Any]) -> MetricResult:
        #compute bus factor score based on contributor analysis
        with measure_time() as get_latency:
            score = await self._calculate_bus_factor_score(context, config)
        
        return MetricResult(score=score, latency=get_latency())
    
    async def _calculate_bus_factor_score(self, context: ModelContext, config: Dict[str, Any]) -> float:
        """Calculate bus factor based on unique commit authors: min(1.0, contributors / 5.0)."""
        contributors = 0
        
        # Try to get contributor count from git repository analysis
        if context.code_repos:
            git_inspector = GitInspector()
            try:
                for code_repo in context.code_repos:
                    repo_path = git_inspector.clone_repo(code_repo)
                    if repo_path:
                        analysis = git_inspector.analyze_repository(repo_path)
                        contributor_data = analysis.get('contributor_analysis', {})
                        
                        # Get unique authors (prefer last 12 months, otherwise all-time)
                        recent_authors = contributor_data.get('recent_unique_authors', 0)
                        all_time_authors = contributor_data.get('unique_authors', 0)
                        
                        contributors = recent_authors if recent_authors > 0 else all_time_authors
                        break  # Use first available repo
            finally:
                git_inspector.cleanup()
        
        # If no code repos or no contributors found, try to estimate from HF metadata
        if contributors == 0 and context.hf_info:
            # Estimate contributors based on HF engagement as fallback
            downloads = context.hf_info.get('downloads', 0)
            likes = context.hf_info.get('likes', 0)
            
            # Rough estimation: high engagement suggests more contributors
            if downloads > 100000 and likes > 100:
                contributors = 5  # Assume well-maintained project
            elif downloads > 10000 and likes > 50:
                contributors = 3
            elif downloads > 1000 and likes > 10:
                contributors = 2
            else:
                contributors = 1
        
        # Apply specification formula: BusFactor = min(1.0, contributors / 5.0)
        return min(1.0, contributors / 5.0)
    
    #analyze hugging face engagement
    def _analyze_hf_engagement(self, hf_info: Dict[str, Any]) -> float:
        downloads = hf_info.get('downloads', 0)
        likes = hf_info.get('likes', 0)
        
        #score based on community engagement
        engagement_score = 0.0
        
        #downloads contribution
        if downloads > 10000:
            engagement_score += 0.4
        elif downloads > 1000:
            engagement_score += 0.3
        elif downloads > 100:
            engagement_score += 0.2
        elif downloads > 10:
            engagement_score += 0.1
        
        #likes contribution  
        if likes > 100:
            engagement_score += 0.3
        elif likes > 50:
            engagement_score += 0.2
        elif likes > 10:
            engagement_score += 0.1
        elif likes > 0:
            engagement_score += 0.05
        
        #recent activity
        if hf_info.get('last_modified'):
            #add better date parsing
            engagement_score += 0.1
        
        return min(0.8, engagement_score)  #cap hugging face only score
