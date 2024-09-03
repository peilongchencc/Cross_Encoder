from sentence_transformers import CrossEncoder

class CustomCrossEncoder():
    def __init__(self):
        self.model = CrossEncoder(model_name="BAAI/bge-reranker-large")
        
    def rank_documents(self, query, documents, num_workers=0):
        """使用CrossEncoder模型对文档进行评分并返回排序后的文档列表。
        """
        scores_rank_list = self.model.rank(query, documents, return_documents=True, num_workers=num_workers)
        
        # 将np.float32转换为标准float
        for item in scores_rank_list:
            # 使用round只保留小数点后5位
            item['score'] = round(float(item['score']), 5)
            
        return scores_rank_list
