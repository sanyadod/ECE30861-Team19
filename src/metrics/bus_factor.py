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
        """Calculate bus factor based on git analysis and HF metadata."""
        thresholds = config.get('thresholds', {}).get('bus_factor', {})
        min_contributors = thresholds.get('min_contributors', 3)
        recent_window_days = thresholds.get('recent_commits_window_days', 90)
        single_contributor_penalty = thresholds.get('single_contributor_penalty', 0.5)
        
        total_score = 0.0
        weight_hf = 0.6  #hugging face based analysis
        weight_git = 0.4  #git based analysis
        
        #hugging Face based analysis
        if context.hf_info:
            hf_score = self._analyze_hf_engagement(context.hf_info)
            total_score += hf_score * weight_hf
        else:
            #increase git weight if no hugging face
            weight_git = 1.0
        
        #git repo analysis
        git_score = 0.0
        if context.code_repos:
            git_inspector = GitInspector()
            try:
                for code_repo in context.code_repos:
                    repo_path = git_inspector.clone_repo(code_repo)
                    if repo_path:
                        analysis = git_inspector.analyze_repository(repo_path)
                        contributor_data = analysis.get('contributor_analysis', {})
                        
                        unique_authors = contributor_data.get('unique_authors', 0)
                        
                        #score based on contributor count
                        if unique_authors >= min_contributors:
                            git_score = 0.8
                        elif unique_authors == 2:
                            git_score = 0.6
                        elif unique_authors == 1:
                            git_score = single_contributor_penalty
                        else:
                            git_score = 0.1
                        break  #first available repo
            finally:
                git_inspector.cleanup()
        
        total_score += git_score * weight_git
        
        return min(1.0, total_score)
    
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
