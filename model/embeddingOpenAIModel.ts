import { OpenAIEmbeddings } from "@langchain/openai";

export const embeddingOpenAIModel = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});