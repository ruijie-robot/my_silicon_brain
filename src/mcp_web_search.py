import requests
import json
import asyncio
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("web_search")

class WebSearcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """使用DuckDuckGo搜索"""
        try:
            url = "https://duckduckgo.com/html/"
            params = {
                'q': query,
                'b': '',
                'kl': 'wt-wt'
            }
            
            response = self.session.get(url, params=params)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            result_elements = soup.find_all('div', class_='result')
            
            for i, element in enumerate(result_elements[:max_results]):
                title_elem = element.find('a', class_='result__a')
                snippet_elem = element.find('div', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'url': title_elem.get('href'),
                        'snippet': snippet_elem.get_text(strip=True),
                        'source': 'DuckDuckGo'
                    })
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def get_page_content(self, url: str, max_length: int = 2000) -> str:
        """获取网页内容"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 获取文本内容
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:max_length]
            
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return ""
    
    def search_finance_news(self, query: str = None) -> List[Dict[str, Any]]:
        """搜索金融新闻"""
        news_sources = [
            {
                'name': '新浪财经',
                'rss_url': 'http://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=1686&k=&num=50&page=1&r=0.123'
            },
            {
                'name': '东方财富',
                'rss_url': 'http://feed.eastmoney.com/news/cjzx.xml'
            }
        ]
        
        all_news = []
        
        for source in news_sources:
            try:
                feed = feedparser.parse(source['rss_url'])
                
                for entry in feed.entries[:10]:  # 限制每个源的新闻数量
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    
                    # 如果有查询词，进行过滤
                    if query and query.lower() not in title.lower() and query.lower() not in summary.lower():
                        continue
                    
                    news_item = {
                        'title': title,
                        'url': entry.get('link', ''),
                        'summary': summary,
                        'published': entry.get('published', ''),
                        'source': source['name']
                    }
                    all_news.append(news_item)
                    
            except Exception as e:
                print(f"Error fetching news from {source['name']}: {e}")
                continue
        
        # 按发布时间排序
        all_news.sort(key=lambda x: x.get('published', ''), reverse=True)
        return all_news[:20]  # 返回最新的20条


searcher = WebSearcher()

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    搜索网络内容
    
    Args:
        query: 搜索查询词
        max_results: 最大结果数量 (默认: 5)
        
    Returns:
        搜索结果的JSON字符串
    """
    try:
        results = searcher.search_duckduckgo(query, max_results)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@mcp.tool()
def get_webpage_content(url: str, max_length: int = 2000) -> str:
    """
    获取网页内容
    
    Args:
        url: 网页URL
        max_length: 最大内容长度 (默认: 2000)
        
    Returns:
        网页文本内容
    """
    try:
        content = searcher.get_page_content(url, max_length)
        return content if content else "无法获取网页内容"
    except Exception as e:
        return f"错误: {str(e)}"

@mcp.tool()
def search_finance_news(query: str = None) -> str:
    """
    搜索金融新闻
    
    Args:
        query: 可选的查询关键词，用于过滤新闻
        
    Returns:
        金融新闻列表的JSON字符串
    """
    try:
        news = searcher.search_finance_news(query)
        return json.dumps(news, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@mcp.tool()
def search_stock_info(stock_symbol: str) -> str:
    """
    搜索股票信息
    
    Args:
        stock_symbol: 股票代码或公司名称
        
    Returns:
        股票相关信息的JSON字符串
    """
    try:
        # 搜索股票相关信息
        query = f"{stock_symbol} 股票 行情 财务"
        results = searcher.search_duckduckgo(query, 3)
        
        # 获取一些详细内容
        for result in results:
            if result.get('url'):
                content = searcher.get_page_content(result['url'], 500)
                result['content_preview'] = content
        
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@mcp.tool()
def search_market_sector(sector: str) -> str:
    """
    搜索市场板块信息
    
    Args:
        sector: 板块名称（如：新能源、医药、科技等）
        
    Returns:
        板块相关信息的JSON字符串
    """
    try:
        query = f"{sector} 板块 行情 龙头股 概念股"
        results = searcher.search_duckduckgo(query, 5)
        
        # 同时搜索相关新闻
        news_query = f"{sector} 板块 最新消息"
        news = searcher.search_finance_news(news_query)
        
        return json.dumps({
            "search_results": results,
            "related_news": news[:5]  # 只取前5条新闻
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport='stdio')