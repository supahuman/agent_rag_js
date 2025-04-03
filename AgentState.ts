import { retriever } from './retriever';

import { Annotation } from '@langchain/langgraph';
import { BaseMessage } from '@langchain/core/messages';

import { createRetrieverTool } from "langchain/tools/retriever";
import { ToolNode } from "@langchain/langgraph/prebuilt";

export const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (prev, newMessage) => prev.concat(newMessage),
        default: () => [],
    })
})

const tool = createRetrieverTool(
    retriever,
    {
        name: "retrieve_blog_posts",
        description: "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs."
    }
);

export const tools = [tool];

export const toolNode = new ToolNode<typeof GraphState.State>(tools);