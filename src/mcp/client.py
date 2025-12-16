import asyncio
import httpx

BASE_URL = "http://localhost:8000"

async def main():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=None) as client:
        # Health
        r = await client.get("/health")
        r.raise_for_status()
        print("Health:", r.json())

        # List models
        r = await client.get("/models")
        r.raise_for_status()
        print("Models under test:", r.json())

        # Run QA pipeline (sync)
        r = await client.post("/pipelines/qa")
        r.raise_for_status()
        print("QA pipeline result:", r.json())

        # Run ToolScale pipeline (sync)
        r = await client.post("/pipelines/toolscale")
        r.raise_for_status()
        print("ToolScale pipeline result:", r.json())

        # Optional: run generic pipeline by dataset name
        # r = await client.post("/pipelines/QADataset")
        # r.raise_for_status()
        # print("Generic pipeline result:", r.json())

        # Optional: async fire-and-forget
        # r = await client.post("/pipelines/QADataset/async")
        # r.raise_for_status()
        # print("Async accepted:", r.json())

if __name__ == "__main__":
    asyncio.run(main())