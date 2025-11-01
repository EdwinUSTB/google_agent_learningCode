from fastmcp import FastMCP

mcp_server = FastMCP()

@mcp_server.tool
def greet(name:str) -> str:
    """
    生成一个简单的问候语
    """
    return f"Hello, {name},我是啊贵，很高兴遇到你!"

if __name__ == "__main__":
    mcp_server.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
        debug=True
    )



