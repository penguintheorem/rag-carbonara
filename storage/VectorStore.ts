import { Embeddings } from "@langchain/core/embeddings";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

export class VectorStore {
  private vectorStore: MemoryVectorStore;

  constructor(embeddings: Embeddings) {
    this.vectorStore = new MemoryVectorStore(embeddings);
  }

  getVectorStore() {
    return this.vectorStore;
  }

}