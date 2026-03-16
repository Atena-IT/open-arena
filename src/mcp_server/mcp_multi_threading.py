import asyncio, queue, threading
from src.mcp_server.mcp_sse_bearer import MCPSSEBearer


""" CLASSES """
class MCPWorker:
    """
    Dedicated MCP worker running in its own thread with a private asyncio event loop. The worker opens a persistent MCP
    SSE connection at startup and keeps it alive for the whole lifetime of the thread, exposing 'mcp_session' and
    'mcp_tools' for reuse. 'submit()' lets synchronous code execute coroutines on the worker loop (thread-safe) and wait
    for the result. Call 'close()' to stop the loop and close the SSE connection.
    Parameters:
        :param mcp_url: Server access url for MCP.
        :param token: MCP access token.
        :param startup_timeout: Waiting period before starting.
    """
    def __init__(self, mcp_url: str, token: str, startup_timeout: float = 10.0):
        self.mcp_url = mcp_url
        self.token = token
        self.startup_timeout = startup_timeout

        # Loop
        self.loop = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._startup_error = None

        # MCP
        self.mcp = None
        self.mcp_session = None
        self.mcp_tools = None

        # Thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

        if not self._ready.wait(timeout=self.startup_timeout):
            raise RuntimeError("MCP worker did not start correctly (timeout)")
        if self._startup_error is not None:
            raise RuntimeError("MCP worker failed during initialization") from self._startup_error


    def run(self):
        """
        Thread entrypoint: create loop, init MCP, run forever, then shutdown when stopped.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            async def bootstrap():
                self.mcp = MCPSSEBearer(mcp_url=self.mcp_url, token=self.token)
                await self.mcp.__aenter__()  # keep it open
                self.mcp_session = self.mcp.session
                self.mcp_tools = self.mcp.tools

            self.loop.run_until_complete(bootstrap())
        except Exception as exc:
            self._startup_error = exc
            return
        finally:
            self._ready.set()

        try:
            self.loop.run_forever()
        finally:
            try:
                if self.mcp is not None:
                    self.loop.run_until_complete(self.mcp.__aexit__(None, None, None))
            finally:
                self.loop.close()
                self._closed.set()


    def submit(self, coro, timeout: float | None = None):
        """
        Execute a coroutine on the worker loop and block until completion.
        """
        if self.loop is None or not self._ready.is_set() or self._startup_error is not None:
            raise RuntimeError("MCP worker is not ready")

        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout)


    def close(self, timeout: float = 10.0):
        """
        Stop the loop and close the persistent SSE connection.
        """
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        self._closed.wait(timeout=timeout)


class MCPWorkerPool:
    """
    Fixed-size pool of reusable 'MCPWorker' instances.
    Each `MCPWorker` owns a dedicated thread + asyncio event loop + a persistent MCP SSE connection.
    The pool uses a thread-safe queue to hand out an *available* worker to callers:
    - 'acquire()' blocks if all workers are currently in use (backpressure).
    - 'release()' returns the worker to the pool so it can be reused by other requests.
    Parameters:
        :param mcp_url: MCP server URL used to create each worker connection.
        :param size: Number of workers (and therefore persistent SSE connections) to keep in the pool.
        :param token: Authentication token forwarded to each worker.
    """
    def __init__(self, mcp_url: str, size: int, token: str):
        self.size = size
        self.queue = queue.Queue(maxsize=size)
        self.workers = []
        for _ in range(size):
            w = MCPWorker(mcp_url=mcp_url, token=token)
            self.workers.append(w)
            self.queue.put(w)


    def acquire(self) -> MCPWorker:
        """
        Borrow a worker from the pool. This call blocks until a worker becomes available.
        Returns:
            An MCPWorker instance ready to run tasks.
        """
        return self.queue.get()


    def release(self, worker: MCPWorker):
        """
        Return a previously acquired worker to the pool.
        Parameters:
            :param worker: The MCPWorker to release.
        """
        self.queue.put(worker)


    def close(self):
        """
        Stop all workers and close their underlying MCP SSE connections.
        """
        for w in self.workers:
            w.close()
