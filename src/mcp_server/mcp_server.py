from __future__ import annotations
import logging, os, time, uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi_mcp import AuthConfig, FastApiMCP
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


""" CONFIG """
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


""" MODELS """
class EchoRequest(BaseModel):
    text: str = Field(..., description="Text to echo back.")
    uppercase: bool = Field(False, description="If true, uppercase the output.")


class EchoResponse(BaseModel):
    output: str


class AddRequest(BaseModel):
    a: float
    b: float


class AddResponse(BaseModel):
    result: float


class KPIRequest(BaseModel):
    system: str
    kpi_name: str
    region: Optional[str] = "eu"


class LLMEvaluationGateway:
    """
    Middleware server that exposes FastAPI endpoints and mounts them as MCP tools via fastapi_mcp.
    Usage:
        gateway = LLMEvaluationGateway()
        app = gateway.app
        uvicorn.run(app, host="0.0.0.0", port=9000)
    Parameters:
        :param name: Application name
        :param title: Application title
        :param description: Application Description
        :param version: Application version
        :param protect_mcp_with_token: Enable the token protection
    """
    def __init__(self, *,
                 name: str = "Demo-MCP",
                 title: str = "Demo MCP (fastapi_mcp)",
                 description: str = "Demo server: FastAPI endpoints automatically exposed as MCP tools.",
                 version: str = "0.1.0",
                 protect_mcp_with_token: bool = True) -> None:
        self.app = FastAPI(title=title, description=description, version=version)
        self.name = name
        self.protect_mcp_with_token = protect_mcp_with_token
        if self.protect_mcp_with_token and not os.getenv("MCP_TOKEN", "").strip():
            raise RuntimeError("MCP_TOKEN is required when protect_mcp_with_token=True")
        self.register_routes()
        self.mcp = self.mount_mcp()


    @staticmethod
    def token_auth_scheme(x_mcp_token: Optional[str] = Header(None, alias="X-MCP-Token")) -> str:
        """
        Authentication check method.
        Parameters:
            :param x_mcp_token: Access token provided by the client in header `X-MCP-Token`.
        Returns:
            :return: The validated token string.
        Raises:
            :exception HTTPException: if MCP_TOKEN is missing on server or token is invalid.
        """
        expected = os.getenv("MCP_TOKEN", "").strip()
        if not expected:
            raise HTTPException(status_code=500, detail="MCP_TOKEN not configured on server")
        if not x_mcp_token or x_mcp_token.strip() != expected:
            raise HTTPException(status_code=401, detail="Invalid MCP token")
        return x_mcp_token


    def register_routes(self) -> None:
        """
        MCP tools
        """
        # /echo
        @self.app.post(
            "/echo",
            operation_id="echo",
            response_model=EchoResponse,
            status_code=status.HTTP_200_OK,
            summary="Echo text (optionally uppercase)",
            description="Returns the input text unchanged, or uppercased if `uppercase=true`.",
        )
        def echo(request: EchoRequest) -> EchoResponse:
            out = request.text.upper() if request.uppercase else request.text
            return EchoResponse(output=out)

        # /add
        @self.app.post(
            "/add",
            operation_id="add",
            response_model=AddResponse,
            status_code=status.HTTP_200_OK,
            summary="Add two numbers",
            description="Returns the sum of two given float numbers",
        )
        def add(request: AddRequest) -> AddResponse:
            return AddResponse(result=request.a + request.b)

        # /time
        @self.app.get(
            "/time",
            operation_id="get_unix_time",
            status_code=status.HTTP_200_OK,
            summary="Get unix time",
            description="Returns the unix time",
        )
        def get_unix_time() -> Dict[str, Any]:
            return {"unix_time": int(time.time())}

        # /kpi
        @self.app.post(
            "/kpi",
            operation_id="simulate_kpi_fetch",
            status_code=status.HTTP_200_OK,
            summary="Fake KPI fetch (static data)",
            description="Returns a mocked KPI",
        )
        def simulate_kpi_fetch(request: KPIRequest) -> Dict[str, Any]:
            return self.simulate_kpi_fetch(request)

        # /health
        @self.app.get("/health", operation_id="health", status_code=200)
        def health() -> Dict[str, str]:
            return {"status": "ok"}


    @staticmethod
    def simulate_kpi_fetch(requests: KPIRequest) -> Dict[str, Any]:
        database = {
            ("billing", "success_rate", "eu"): 99.3,
            ("billing", "success_rate", "us"): 98.7,
            ("search", "p95_latency_ms", "eu"): 180,
            ("search", "p95_latency_ms", "us"): 210,
        }
        key = (
            requests.system.lower().strip(),
            requests.kpi_name.lower().strip(),
            (requests.region or "eu").lower().strip(),
        )
        value = database.get(key)
        if value is None:
            return {"found": False, "system": requests.system, "kpi_name": requests.kpi_name, "region": requests.region}
        return {"found": True, "system": requests.system, "kpi_name": requests.kpi_name, "region": requests.region, "value": value}


    def mount_mcp(self) -> Any:
        """
        Mount MCP wrapper on the FastAPI app.
        """
        auth_config = None
        if self.protect_mcp_with_token:
            auth_config = AuthConfig(dependencies=[Depends(self.token_auth_scheme)])
        mcp = FastApiMCP(
            self.app,
            name=self.name,
            description="MCP wrapper for demo endpoints",
            auth_config=auth_config,
            describe_all_responses=True,
            describe_full_response_schema=True,
        )

        # SSE
        return mcp.mount_sse()


""" MAIN """
if __name__ == "__main__":
    gateway = LLMEvaluationGateway(protect_mcp_with_token=True)
    app = gateway.app
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FAST_API_PORT", "9000")))
