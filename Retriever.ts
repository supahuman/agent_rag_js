import dotenv from 'dotenv';
dotenv.config();


// Retriever 
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";

const urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

const docs = await Promise.all(
    urls.map(url => new CheerioWebBaseLoader(url).load())
);

const docsList = docs.flat();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50
});

const docSplits = await textSplitter.splitDocuments(docsList);

// Add to vectorDB
const vectorStore = await MemoryVectorStore.fromDocuments(
    docSplits,
    new OpenAIEmbeddings(),
);

export const retriever = vectorStore.asRetriever();



