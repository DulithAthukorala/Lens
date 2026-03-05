from dotenv import load_dotenv
load_dotenv()

from langfuse import get_client

langfuse = get_client()

with langfuse.start_as_current_observation(as_type="span", name="smoke-test") as span:
    span.update(output="Langfuse trace received ✅")

langfuse.flush()
print("Sent Langfuse smoke-test trace.")