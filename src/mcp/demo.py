from __future__ import annotations
import logging, os, time, uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi_mcp import AuthConfig, FastApiMCP
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


""" CONFIG """
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
        :param protect_endpoints_with_token: Enable the token protection
    """
    def __init__(self, *, name: str = "Demo-MCP", title: str = "Demo MCP (fastapi_mcp)", description: str = "Demo server: FastAPI endpoints automatically exposed as MCP tools.", version: str = "0.1.0", protect_endpoints_with_token: bool = False) -> None:
        self.app = FastAPI(title=title, description=description, version=version)
        self.name = name
        self.protect_endpoints_with_token = protect_endpoints_with_token
        self._register_routes()
        self._mount_mcp()


    @staticmethod
    def token_auth_scheme(x_mcp_token: str = Header(..., alias="X-MCP-Token")) -> str:
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
        if x_mcp_token.strip() != expected:
            raise HTTPException(status_code=401, detail="Invalid MCP token")
        return x_mcp_token


    def _auth_dependency(self):
        """
        Returns the dependency to use on endpoints, or None if auth is disabled.
        """
        if not self.protect_endpoints_with_token:
            return None
        return Depends(self.token_auth_scheme)


    def _register_routes(self) -> None:
        """
        MCP tools
        """
        auth_dep = self._auth_dependency()

        # /echo
        if auth_dep is not None:
            @self.app.post(
                "/echo",
                operation_id="echo",
                response_model=EchoResponse,
                status_code=status.HTTP_200_OK,
                summary="Echo text (optionally uppercase)",
                description="Returns the input text unchanged, or uppercased if `uppercase=true`."
            )
            def echo(req: EchoRequest, token=auth_dep) -> EchoResponse:  # noqa: ARG001
                out = req.text.upper() if req.uppercase else req.text
                return EchoResponse(output=out)
        else:
            @self.app.post(
                "/echo",
                operation_id="echo",
                response_model=EchoResponse,
                status_code=status.HTTP_200_OK,
                summary="Returns the input text unchanged, or uppercased if `uppercase=true`.",
            )
            def echo(req: EchoRequest) -> EchoResponse:
                out = req.text.upper() if req.uppercase else req.text
                return EchoResponse(output=out)

        # /add
        if auth_dep is not None:
            @self.app.post(
                "/add",
                operation_id="add",
                response_model=AddResponse,
                status_code=status.HTTP_200_OK,
                summary="Add two numbers",
                description="Returns the sum of two given float numbers"
            )
            def add(req: AddRequest, token=auth_dep) -> AddResponse:  # noqa: ARG001
                return AddResponse(result=req.a + req.b)
        else:
            @self.app.post(
                "/add",
                operation_id="add",
                response_model=AddResponse,
                status_code=status.HTTP_200_OK,
                summary="Add two numbers",
                description="Returns the sum of two given float numbers"
            )
            def add(req: AddRequest) -> AddResponse:
                return AddResponse(result=req.a + req.b)

        # /time
        if auth_dep is not None:
            @self.app.get(
                "/time",
                operation_id="get_unix_time",
                status_code=status.HTTP_200_OK,
                summary="Get unix time",
                description="Returns the unix time"
            )
            def get_unix_time(token=auth_dep) -> Dict[str, Any]:  # noqa: ARG001
                return {"unix_time": int(time.time())}
        else:
            @self.app.get(
                "/time",
                operation_id="get_unix_time",
                status_code=status.HTTP_200_OK,
                summary="Get unix time",
                description="Returns the unix time"
            )
            def get_unix_time() -> Dict[str, Any]:
                return {"unix_time": int(time.time())}

        # /kpi
        if auth_dep is not None:
            @self.app.post(
                "/kpi",
                operation_id="simulate_kpi_fetch",
                status_code=status.HTTP_200_OK,
                summary="Fake KPI fetch (static data)",
                description="Returns a mocked KPI",
            )
            def simulate_kpi_fetch(req: KPIRequest, token=auth_dep) -> Dict[str, Any]:  # noqa: ARG001
                return self._simulate_kpi_fetch(req)
        else:
            @self.app.post(
                "/kpi",
                operation_id="simulate_kpi_fetch",
                status_code=status.HTTP_200_OK,
                summary="Fake KPI fetch (static data)",
                description="Returns a mocked KPI",
            )
            def simulate_kpi_fetch(req: KPIRequest) -> Dict[str, Any]:
                return self._simulate_kpi_fetch(req)

        # /health (di solito non protetta)
        @self.app.get("/health", operation_id="health", status_code=200)
        def health() -> Dict[str, str]:
            return {"status": "ok"}


    @staticmethod
    def _simulate_kpi_fetch(req: KPIRequest) -> Dict[str, Any]:
        db = {
            ("billing", "success_rate", "eu"): 99.3,
            ("billing", "success_rate", "us"): 98.7,
            ("search", "p95_latency_ms", "eu"): 180,
            ("search", "p95_latency_ms", "us"): 210
        }
        key = (req.system.lower().strip(), req.kpi_name.lower().strip(), (req.region or "eu").lower().strip())
        value = db.get(key)

        if value is None:
            return {"found": False, "system": req.system, "kpi_name": req.kpi_name, "region": req.region}

        return {"found": True, "system": req.system, "kpi_name": req.kpi_name, "region": req.region, "value": value}


    def _mount_mcp(self) -> None:
        """
        Mount MCP wrapper on the FastAPI app.
        """
        # AuthConfig in your version requires dependencies/issuer/custom_oauth_metadata.
        # We use dependencies for a simple header token check.
        auth_config = None
        if self.protect_endpoints_with_token:
            auth_config = AuthConfig(dependencies=[Depends(self.token_auth_scheme)])

        mcp = FastApiMCP(
            self.app,
            name=self.name,
            description="MCP wrapper for demo endpoints",
            auth_config=auth_config,
            describe_all_responses=True,
            describe_full_response_schema=True,
        )
        # In your sample code you used mount(), but earlier you used mount_http().
        # Keep the same method you use in your environment. Here we follow your sample: mount().
        mcp.mount()
        self.mcp = mcp  # keep reference if you need it later (optional)


""" ENTRYPOINT """
load_dotenv()
gateway = LLMEvaluationGateway()
app = gateway.app
for r in app.routes:
    methods = getattr(r, "methods", None)
    name = getattr(r, "name", "")
    print(f"{getattr(r,'path','')}  {sorted(list(methods)) if methods else ''}  name={name}")


""" MAIN """
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
