"""
长期记忆的实现
"""
from langgraph.store.memory import InMemoryStore

def embed(texts:list[str]) -> list[list[float]]:
    return [[1.0,2.0] for _ in texts]

store = InMemoryStore(index={"embed":embed,"dims":2})
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id,application_context)


store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "用户喜欢简短直接的语言",
            "用户只说中文和python",
        ],
        "my-key": "my-value",
    },
)

item = store.get(namespace, "a-memory")
print("检索结果：",item)

items = store.search(
    namespace, 
    filter={"my-key":"my_value"},
    query="语言偏好"
)

print("搜索结果：",items)