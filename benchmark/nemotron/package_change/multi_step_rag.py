# motiongreactor/orchestrators/multi_step_rag.py

import asyncio
from typing import Iterable

from aiwrappifymodels.message import MessageForLLM

from motiongreactor.embeddings.base_embedding import BaseEmbeddingService

from ..context_providers.base_context_provider import BaseContextProvider
from ..llm.base_llm import BaseLLM
from ..orchestrators.base import BaseOrchestrator
from ..rerankers.base_reranker import BaseReranker
import re

class MultiStepRagOrchestrator(BaseOrchestrator):
    """
    what does refine do?

    original question and subquery Q & A

    suppose we split into, and we have Q0
    ["Q1", "Q2", "Q3"]

    now we do the first round of C1 = hybrid_search(Q1) , but, sometimes, where is no need to do this,
    we can just let llm answer, without any context, and get the A, then refine


    then, use C1 and Q1 to get A1 = answer(C1, Q1)

    from A1, we start to refine the remaining subqueries .
    we pass (  ["Q1", "Q2", "Q3"], Q0, A1 ) to refine function and expect output to be ["Q2'", "Q3'"],
    hopefully it become less

    and finaly we will hopefully end up with ["Q3'"]

    then we will do the final round of C3 = hybrid_search(Q3')
    A3 = answer(C3, Q3)
    """

    def __init__(
        self,
        llm: BaseLLM,
        context_providers: list[BaseContextProvider],
        reranker: BaseReranker,
        embedding: BaseEmbeddingService,
    ):
        self.llm = llm
        self.context_providers = context_providers
        self.reranker = reranker
        self.embedding = embedding

    async def hybrid_search(self, query: str) -> list[str]:
        query_vector = self.embedding.embed([query])
        contexts = []
        tasks = [
            context_provider.provide_contexts(query_vector,query)
            for context_provider in self.context_providers
        ]
        results = await asyncio.gather(*[task for task in tasks])
        for result in results:
            for chunk in result:
                contexts.append(chunk)

        if len(contexts) == 0:
            return []

        best_context = self.reranker.rerank(contexts, query, 3)  ## top 3
        return best_context

    # def web_search(self, query: str, k: int = 3) -> list[str]: ...

    def answer_a_sub_query(
        self, sub_query: str, conversation: list[MessageForLLM]
    ) -> str:
        contexts = asyncio.run(self.hybrid_search(sub_query))
        if contexts:
            prompt = f"""
            Here are the contexts that can help you answer, please choose one or none of them: \n\n{contexts}
            """
        else:
            prompt = f"""
            There are no contexts that can help you answer this question, please provide the answer.
            """
        # import ipdb; ipdb.set_trace()
        conversation[-1].content = sub_query
        
        return self.llm.full_response(conversation, prompt, 0)

    def split_query(self, unmodified_query_from_user: str) -> list[str]:
        sub_query_prefix = "sub-query"
        sys_prompt = f"""   
        Your answer format:
        {sub_query_prefix} 1: your sub question
        {sub_query_prefix} 2: your sub question
        {sub_query_prefix} 3: your sub question

        **Example Output:**  
        some resoning regarding the breakdown
        <START>
        {sub_query_prefix} 1: What is the capital of France?  
        {sub_query_prefix} 2: What is the population of Paris?  
        <END>

        You are an intelligent assistant specialized in solving multi-hop questions. 
        Split the query into smaller sub-queries, ensuring each sub-question is specific and answerable with only one entity.
        must follow the format above without other information, otherwise, the system will not be able to understand.
        """

        parts = self.llm.get_response(
            sys_prompt=sys_prompt, user_prompt=unmodified_query_from_user, temperature=0
        )#.split("\n")

        # print(parts)
        parts = re.search(r"<START>(.*?)<END>", parts, re.DOTALL).group(1).strip().split("\n")
        # print("After regex")

        return [
            part.split(":")[-1].strip()
            for part in parts
            if part.startswith(sub_query_prefix)
        ]

    def refine(
        self, sub_queries: list[str], observations: str, original_query: str
    ) -> list[str]:

        # observation is the Q A pair of all the previous subqueries
        # sub_query_prefix = "sub-query"
        # prompt = f"""
        #     Take a deep breath, and read the requirements carefully.
        #     Your answer format MUST FOLLOW:
        
        #     {sub_query_prefix} 1: your sub question
        #     {sub_query_prefix} 2: your sub question
        #     {sub_query_prefix} 3: your sub question


        #     You are an intelligent assistant specialized in solving multi-hop questions.
        #     Your task is to refine the subquery list with the observed answer of sub questions.
        #     MUST follow the format above without other information, otherwise, the system will not be able to understand.
        #     Your observations are:
        #     {observations}
        #     The final question is:
        #     {original_query}
        #     Follow these steps to refine the sub queries:
        #     1. Use the answer you got from one sub query to replace the keywords in the other subquestion.
        #     2. Remove the sub query that already answered.

        #     must!!! follow the format above without other information, otherwise, the system will not be able to understand.
        # """
        sub_query_prefix = "sub-query"
        prompt = f"""
        Your answer format:
        {sub_query_prefix} 1: your sub question
        {sub_query_prefix} 2: your sub question
        {sub_query_prefix} 3: your sub question

        You are an intelligent assistant specialized in solving multi-hop questions.  
        Your task is to refine the subquery list using the observed answers to the sub-questions.  
        Strictly follow the format above for your output.  

        **Steps to refine the sub-queries:**  
        1. Use the answer from one sub-query to replace keywords in other sub-queries, if applicable.  
        2. Remove any sub-query that has already been fully answered.  

        **Key Requirements:**  
        1. Your output MUST strictly follow the format. 
        2. Number each sub-query sequentially and ensure continuity in numbering.

        **Input Information:**  
        - Observations can reference.:  
        {observations}  

        - Final Question:  
        {original_query}  

        **Example Output:**  
        some resoning regarding the breakdown
        <START>
        {sub_query_prefix} 1: What is the capital of France?  
        {sub_query_prefix} 2: What is the population of Paris?  
        <END>
        Any 
        
        If your output does not follow the format, the system will not be able to understand it.
        You are allowed to add any briefly reasoning or explanation before the <START> and after <END>.So please make sure to follow the format.
        keep your output clean and clear and brief.
        """

        user_prompt = "\n".join(sub_queries)

        parts = self.llm.get_response(
            sys_prompt=prompt, user_prompt=user_prompt, temperature=0
        )#.split("\n")


        print(parts)

        parts = re.search(r"<START>(.*?)<END>", parts, re.DOTALL).group(1).strip().split("\n")
        # print("After regex")
        print(parts)

        return [
            part.split(":")[-1].strip()
            for part in parts
            if part.startswith(sub_query_prefix)
        ]

    def summarize(
        self,
        unmodified_query_from_user: str,
        conversation: list[MessageForLLM],
        observations: dict | str,
    ) -> Iterable[str]:
        sys_prompt = f"""
        You are an intelligent assistant specialized in solving multi-hop questions.
        Your task is to chat with user with the observations you have.
        
        obersevations:
        {observations}
        """
        print("finally sys prompt is:")
        print(sys_prompt)

        # restore the conversation
        conversation[-1].content = unmodified_query_from_user

        # generate product level prompt
        try:
            response = self.llm.stream_response(
                conversation=conversation, sys_prompt=sys_prompt, temperature=0
            )
            for fragment in response:
                if fragment:
                    yield fragment

        except Exception as e:
            # please handle the exception outside
            raise f"An error occurred: {e}"

    def react(
        self, conversation: list[MessageForLLM], product_related_prompt: str = None
    ) -> Iterable[str]:

        unmodified_query_from_user = conversation[-1].content

        print("unmodified query from user is:")
        print(unmodified_query_from_user)

        observations = ""
        sub_queries = self.split_query(unmodified_query_from_user)
        
        print("sub queries are:")
        print(sub_queries)

        max_iterations = len(sub_queries)
        iterations = 0
        while len(sub_queries) and iterations < max_iterations:
            sub_query = sub_queries.pop(0)
            sub_ans = self.answer_a_sub_query(sub_query, conversation)
            observations += sub_query + ":\n" + sub_ans + "\n\n"
            sub_queries = self.refine(
                sub_queries, observations, unmodified_query_from_user
            )
            iterations += 1

        return self.summarize(unmodified_query_from_user, conversation, observations)
