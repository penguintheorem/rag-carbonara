import { ChatOpenAI } from "@langchain/openai";

export const chatOpenAIModel = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0
});