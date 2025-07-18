import "dotenv/config";
import { VectorStore } from './../storage/VectorStore'
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { embeddingOpenAIModel } from "../model/embeddingOpenAIModel";
import { chatOpenAIModel } from '../model/ChatOpenAIModel';

console.log(`OPENAI_API_KEY=${process.env.OPENAI_API_KEY}`);

// Load contents of blog
const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://www.recipetineats.com/carbonara/#h-ingredients-in-carbonara-sauce",
  {
    selector: pTagSelector
  }
);
const docs = await cheerioLoader.load();
console.assert(docs.length === 1);
console.log(`Total characters: ${docs[0].pageContent.length}`);

// Split contents of blog into chunks
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000, chunkOverlap: 200
});
const allSplits = await splitter.splitDocuments(docs);
console.log(`Split blog post into ${allSplits.length} sub-documents.`);


// **STATE**

// Convert chunks then documents to embeddings and add them to the vector store
const vectorStore = new VectorStore(embeddingOpenAIModel);
// addDocuments also converts the documents to embeddings and adds them to the vector store
// await vectorStore.getVectorStore().addDocuments([])
await vectorStore.getVectorStore().addDocuments(allSplits)

// Define prompt template for question-answering (Well-tested RAG prompt)
const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");
// Define state for application
const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

// **NODES** (application steps)

/**
 * Retrieve the most relevant documents from the vector store given an input question
 */
const retrieve = async (state: typeof InputStateAnnotation.State) => {
  const retrievedDocs = await vectorStore.getVectorStore().similaritySearch(state.question)
  return { context: retrievedDocs };
};


/**
 * Generate an answer to the input question using the retrieved documents (state in input contains the retrieved documents)
 * The input state has the context (retrievedContext from the retrieve function)
 */
const generate = async (state: typeof StateAnnotation.State) => {
  // Assemble the model context into a single big string
  const docsContent = state.context.map(doc => doc.pageContent).join("\n");
  // Invoke the prompt template with the question and the context
  // Input = question string and context
  // I think this step prepares the prompt for the chat model using a pre-defined prompt template
  const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
  // Invoke the chat model with the messages
  const response = await chatOpenAIModel.invoke(messages);
  // Get the answer from the chat model
  return { answer: response.content };
};


// **CONTROL FLOW**

// Compile application and test
const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  // Output of Node retrieve is the input of Node generate
  .addNode("generate", generate)
  // Start with the input question
  .addEdge("__start__", "retrieve")
  // Output of Node retrieve is the input of Node generate
  .addEdge("retrieve", "generate")
  // Output of Node generate is the output of the graph
  .addEdge("generate", "__end__")
  .compile();

// **RUN**

// Run the graph with the input question
// const result = await graph.invoke({ question: "What is the main idea of the blog post?" });
// console.log(result);
let inputs = { question: "How to obtain a creamy sauce while making a carbonata?" };

const result = await graph.invoke(inputs);
console.log(`Full context=${JSON.stringify(result.context)}`);
console.log(`\nAnswer: ${result["answer"]}`);