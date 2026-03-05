import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


# -----------------------------
# 1) Prepare schema + client
# -----------------------------
schema_json = DiagnosticReview.model_json_schema()

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

MODEL_NAME = "gpt-oss-120b"

# 并发数：根据你 GPU/服务端情况调（比如 4/8/16/32）
# 太大可能导致排队/超时；太小吞吐不够
CONCURRENCY = 8

# 每条最多重试次数
MAX_ATTEMPTS = 5

# 失败重试的冷却（秒），可以做指数退避
BASE_BACKOFF = 1.0


# -----------------------------
# 2) One async call with retries
# -----------------------------
async def call_one(
    idx: int,
    input_text: str,
    semaphore: asyncio.Semaphore,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns:
      (idx, parsed_json_or_none, error_message_or_none)
    """
    inp = prompt + input_text

    async with semaphore:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": inp}],
                    extra_body={
                        "guided_json": schema_json,
                        "guided_decoding_backend": "outlines",
                    },
                    temperature=0.7,
                    max_tokens=2048,
                )

                content = resp.choices[0].message.content or ""
                parsed = json.loads(content)

                print(f"Row {idx} succeeded on attempt {attempt}")
                return idx, parsed, None

            except Exception as e:
                err = f"Row {idx} failed attempt {attempt}: {e}"
                print(err)

                if attempt < MAX_ATTEMPTS:
                    # 指数退避：1,2,4,8...
                    backoff = BASE_BACKOFF * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                else:
                    print(f"Row {idx} failed after {MAX_ATTEMPTS} attempts.")
                    return idx, None, str(e)


# -----------------------------
# 3) Batch runner (keeps order)
# -----------------------------
async def run_all_async(df_filtered) -> Tuple[List[Optional[Dict[str, Any]]], List[Optional[str]]]:
    semaphore = asyncio.Semaphore(CONCURRENCY)

    tasks = []
    for i in range(df_filtered.shape[0]):
        tasks.append(
            asyncio.create_task(
                call_one(i, df_filtered.iloc[i].input_text, semaphore)
            )
        )

    results = await asyncio.gather(*tasks)

    # 保持与原df行顺序一致
    outputs: List[Optional[Dict[str, Any]]] = [None] * df_filtered.shape[0]
    errors: List[Optional[str]] = [None] * df_filtered.shape[0]

    for idx, parsed, err in results:
        outputs[idx] = parsed
        errors[idx] = err

    return outputs, errors


# -----------------------------
# 4) Execute in notebook
# -----------------------------
# 在 Jupyter 里如果已经有 event loop 运行，直接 asyncio.run 可能报错
# 这里做一个兼容写法：
def run_async_in_notebook(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Jupyter 常见情况：返回一个 Task，让你 await 它
        return asyncio.create_task(coro)
    else:
        return asyncio.run(coro)


# 运行：
task_or_result = run_async_in_notebook(run_all_async(df_filtered))

# 如果返回的是 Task（Jupyter 场景），你需要在下一行：
# responses, errors = await task_or_result
# 如果直接返回结果（非Jupyter/没有running loop），就是：
# responses, errors = task_or_result
