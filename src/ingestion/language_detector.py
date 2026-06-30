EXTENSION_TO_LANGUAGE = {
    ".py":    "python",
    ".js":    "javascript",
    ".ts":    "typescript",
    ".jsx":   "javascript",
    ".tsx":   "typescript",
    ".java":  "java",
    ".go":    "go",
    ".rs":    "rust",
    ".cpp":   "cpp",
    ".c":     "c",
    ".cs":    "csharp",
    ".rb":    "ruby",
    ".php":   "php",
    ".swift": "swift",
    ".kt":    "kotlin",
}


def detect_language(file_extension: str) -> str:
    """Return language name for a given file extension."""
    return EXTENSION_TO_LANGUAGE.get(file_extension.lower(), "unknown")


def is_supported(file_extension: str) -> bool:
    """Return True if this extension is supported."""
    return file_extension.lower() in EXTENSION_TO_LANGUAGE