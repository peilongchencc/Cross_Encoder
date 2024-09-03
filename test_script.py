import aiohttp
import asyncio

# 如果从外网访问，记得修改域名和主程序启动方式。
base_url = "http://127.0.0.1:8800"

async def test_read_root(session):
    async with session.get(f"{base_url}/") as response:
        result = await response.json()
        print("GET / response:")
        print(result)

async def test_rank_documents(session):
    payload = {
        "query": "谁写的《杀死一只知更鸟》？",
        "documents": [
            "《杀死一只知更鸟》是哈珀·李于1960年出版的一部小说。它立即获得成功，赢得了普利策奖，并已成为现代美国文学的经典作品。",
            "哈珀·李是一位美国小说家，以她的小说《杀死一只知更鸟》而闻名。她于1926年出生在阿拉巴马州的蒙罗维尔，并于1961年获得了普利策小说奖。",
            "《白鲸》小说由赫尔曼·梅尔维尔创作，首次出版于1851年。它被认为是美国文学的杰作，涉及痴迷、复仇以及善恶冲突的复杂主题。",
            "简·奥斯汀是英国小说家，主要以她的六部主要小说闻名，这些小说解读、批判并评论了18世纪末英国的乡绅阶层。",
            "《哈利·波特》系列包括七部由英国作家J.K.罗琳撰写的奇幻小说，是现代最受欢迎和好评的书籍之一。",
            "《了不起的盖茨比》是美国作家F·司各特·菲茨杰拉德于1925年出版的一部小说。故事发生在爵士时代，讲述了富翁杰伊·盖茨比追求黛西·布坎南的生活。"
        ]
    }

    async with session.post(f"{base_url}/rank_documents/", json=payload) as response:
        result = await response.json()
        print("POST /rank_documents/ response:")
        for doc in result["ranked_documents"]:
            print(f"corpus_id: {doc['corpus_id']}, score: {doc['score']}, text: {doc['text']}")

async def main():
    async with aiohttp.ClientSession() as session:
        await test_read_root(session)
        await test_rank_documents(session)

if __name__ == "__main__":
    asyncio.run(main())
