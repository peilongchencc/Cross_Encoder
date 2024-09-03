"""
Description: 多进程版本。
Notes: 
每次创建一个新的进程，都需要额外的资源和时间开销。特别是在spawn模式下，Python 会为每个子进程重新启动解释器并导入所需的模块。

这种开销在任务比较轻量或数据量不大时，可能会抵消掉多进程带来的性能提升。

建议:

针对大量数据再使用多进程版本。
"""
import torch.multiprocessing as mp
from sentence_transformers import CrossEncoder

# 设置进程启动方式为 'spawn'
mp.set_start_method('spawn', force=True)

if __name__ == '__main__':
    # 本地加载 BAAI/bge-reranker-large 模型(默认使用CUDA)
    model = CrossEncoder("BAAI/bge-reranker-large")

    query = "谁写的《杀死一只知更鸟》？"
    documents = [
        "《杀死一只知更鸟》是哈珀·李于1960年出版的一部小说。它立即获得成功，赢得了普利策奖，并已成为现代美国文学的经典作品。",
        "《白鲸》小说由赫尔曼·梅尔维尔创作，首次出版于1851年。它被认为是美国文学的杰作，涉及痴迷、复仇以及善恶冲突的复杂主题。",
        "哈珀·李是一位美国小说家，以她的小说《杀死一只知更鸟》而闻名。她于1926年出生在阿拉巴马州的蒙罗维尔，并于1961年获得了普利策小说奖。",
        "简·奥斯汀是英国小说家，主要以她的六部主要小说闻名，这些小说解读、批判并评论了18世纪末英国的乡绅阶层。",
        "《哈利·波特》系列包括七部由英国作家J.K.罗琳撰写的奇幻小说，是现代最受欢迎和好评的书籍之一。",
        "《了不起的盖茨比》是美国作家F·司各特·菲茨杰拉德于1925年出版的一部小说。故事发生在爵士时代，讲述了富翁杰伊·盖茨比追求黛西·布坎南的生活。"
    ]

    scores_rank_list = model.rank(query, documents, return_documents=True, num_workers=10)
    for item in scores_rank_list:
        # 将np.float32转换为标准float
        item['score'] = float(item['score'])
        print(item)