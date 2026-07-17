# Custom routes package
from .categories import categories_bp
from .feedback import feedback_bp
from .proxy_source import proxy_source_bp

__all__ = ["categories_bp", "feedback_bp", "proxy_source_bp"]
