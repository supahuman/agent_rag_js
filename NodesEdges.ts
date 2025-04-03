import { GraphState } from "./AgentState";
import { toolNode } from "./AgentState";
import { tools } from "./AgentState";

import { END } from "@langchain/langgraph";
import { pull } from "langchain/hub";
import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { AIMessage, BaseMessage } from "@langchain/core/messages";

/**
 * Decides whether the agent should retrieve more information or end the process.
 * This function checks the last message in the state for a function call. If a tool call is
 * present, the process continues to retrieve information. Otherwise, it ends the process.
 * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
 * @returns {string} - A decision to either "continue" the retrieval process or "end" it.
 */
export function shouldRetrieve(state: typeof GraphState.State): string {
  const { messages } = state;
  console.log("---DECIDE TO RETRIEVE---");
  const lastMessage = messages[messages.length - 1];

  if ("tool_calls" in lastMessage && Array.isArray(lastMessage.tool_calls) && lastMessage.tool_calls.length) {
    console.log("---DECISION: RETRIEVE---");
    return "retrieve";
  }
  // If there are no tool calls then we finish.
  return END;
}

/**
 * Determines whether the Agent should continue based on the relevance of retrieved documents.
 * This function checks if the last message in the conversation is of type FunctionMessage, indicating
 * that document retrieval has been performed. It then evaluates the relevance of these documents to the user's
 * initial question using a predefined model and output parser. If the documents are relevant, the conversation
 * is considered complete. Otherwise, the retrieval process is continued.
 * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
 * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state with the new message added to the list of messages.
 */
export async function gradeDocuments(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  console.log("---GET RELEVANCE---");

  const { messages } = state;
  const tool = {
    name: "give_relevance_score",
    description: "Give a relevance score to the retrieved documents.",
    schema: z.object({
      binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
    })
  }

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing relevance of retrieved docs to a user question.
  Here are the retrieved docs:
  \n ------- \n
  {context} 
  \n ------- \n
  Here is the user question: {question}
  If the content of the docs are relevant to the users question, score them as relevant.
  Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
  Yes: The docs are relevant to the question.
  No: The docs are not relevant to the question.`,
  );

  const model = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
  }).bindTools([tool], {
    tool_choice: tool.name,
  });

  const chain = prompt.pipe(model);

  const lastMessage = messages[messages.length - 1];

  const score = await chain.invoke({
    question: messages[0].content as string,
    context: lastMessage.content as string,
  });

  return {
    messages: [score]
  };
}

/**
 * Check the relevance of the previous LLM tool call.
 *
 * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
 * @returns {string} - A directive to either "yes" or "no" based on the relevance of the documents.
 */
export function checkRelevance(state: typeof GraphState.State): string {
  console.log("---CHECK RELEVANCE---");

  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  if (!("tool_calls" in lastMessage)) {
    throw new Error("The 'checkRelevance' node requires the most recent message to contain tool calls.")
  }
  const toolCalls = (lastMessage as AIMessage).tool_calls;
  if (!toolCalls || !toolCalls.length) {
    throw new Error("Last message was not a function message");
  }

  if (toolCalls[0].args.binaryScore === "yes") {
    console.log("---DECISION: DOCS RELEVANT---");
    return "yes";
  }
  console.log("---DECISION: DOCS NOT RELEVANT---");
  return "no";
}

// Nodes

/**
 * Invokes the agent model to generate a response based on the current state.
 * This function calls the agent model to generate a response to the current conversation state.
 * The response is added to the state's messages.
 * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
 * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state with the new message added to the list of messages.
 */
export async function agent(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  console.log("---CALL AGENT---");

  const { messages } = state;
  // Find the AIMessage which contains the `give_relevance_score` tool call,
  // and remove it if it exists. This is because the agent does not need to know
  // the relevance score.
  const filteredMessages = messages.filter((message) => {
    if ("tool_calls" in message && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      return message.tool_calls[0].name !== "give_relevance_score";
    }
    return true;
  });

  const model = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    streaming: true,
  }).bindTools(tools);

  const response = await model.invoke(filteredMessages);
  return {
    messages: [response],
  };
}

/**
 * Transform the query to produce a better question.
 * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
 * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state with the new message added to the list of messages.
 */
export async function rewrite(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  console.log("---TRANSFORM QUERY---");

  const { messages } = state;
  const question = messages[0].content as string;
  const prompt = ChatPromptTemplate.fromTemplate(
    `Look at the input and try to reason about the underlying semantic intent / meaning. \n 
Here is the initial question:
\n ------- \n
{question} 
\n ------- \n
Formulate an improved question:`,
  );

  // Grader
  const model = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    streaming: true,
  });
  const response = await prompt.pipe(model).invoke({ question });
  return {
    messages: [response],
  };
}

/**
 * Generate answer
 * @param {typeof GraphState.State} state - The current state of the agent, including all messages.
 * @returns {Promise<Partial<typeof GraphState.State>>} - The updated state with the new message added to the list of messages.
 */
export async function generate(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE---");

  const { messages } = state;
  const question = messages[0].content as string;
  // Extract the most recent ToolMessage
  const lastToolMessage = messages.slice().reverse().find((msg) => msg._getType() === "tool");
  if (!lastToolMessage) {
    throw new Error("No tool message found in the conversation history");
  }

  const docs = lastToolMessage.content as string;

  const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");

  const llm = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    streaming: true,
  });

  const ragChain = prompt.pipe(llm);

  const response = await ragChain.invoke({
    context: docs,
    question,
  });

  return {
    messages: [response],
  };
}