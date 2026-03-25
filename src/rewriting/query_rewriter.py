class QueryRewriter:
    """Minimal QueryRewriter stub used by main.py.

    Methods
    -------
    rewrite(query: str) -> list
        Return a list of expanded/rewritten queries (stub returns the original query in a list).
    """

    def __init__(self, method: str = "default") -> None:
        self.method = method

    def rewrite(self, query: str):
        """Return a list of queries (stub).
        """
        return [query]
