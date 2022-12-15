from typing import Any

from pydantic import BaseModel


class ColdStartProxy:
    def __int__(self, proxy_url: str):
        self.proxy_url = proxy_url
        self.proxy_timeout = 20

    async def handle_request(self, request: BaseModel) -> Any:
        pass
