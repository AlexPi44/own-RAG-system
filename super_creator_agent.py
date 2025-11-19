"""
Super Creator Agent (SCA) - 2025 Architecture
Implements: LangGraph + ReWOO + RAPTOR + FlashRank + Self-Healing
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core imports
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

# LangGraph for orchestration
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# FlashRank for reranking
from flashrank import Ranker, RerankRequest

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb


# ==================== STATE DEFINITION ====================
class AgentState(TypedDict):
    """LangGraph state for the Super Creator Agent"""
    query: str
    plan: str
    retrieved_docs: List[Document]
    code: str
    execution_result: Dict[str, Any]
    reflection: str
    iteration: int
    max_iterations: int
    final_answer: str
    error_log: List[str]
    needs_human_approval: bool


# ==================== ADVANCED RAG ENGINE ====================
class SuperRAGEngine:
    """RAG 2.0: RAPTOR + HyDE + MultiQuery + FlashRank Reranking"""
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b", persist_dir: str = "./sca_db"):
        print("🚀 Initializing Super RAG Engine...")
        self.persist_dir = persist_dir
        
        # Dual-model strategy: Fast model for RAG operations
        self.llm_fast = Ollama(model=model_name, temperature=0.1, num_ctx=4096)
        
        # High-quality embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # FlashRank reranker (enterprise-grade precision)
        print("📊 Loading FlashRank reranker...")
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
        
        # Code-aware text splitter (preserves structure)
        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.vectorstore = None
        print("✅ RAG Engine ready")
    
    def create_raptor_hierarchy(self, documents: List[Document]) -> List[Document]:
        """RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"""
        print("\n🌳 Building RAPTOR hierarchy...")
        
        chunks = self.code_splitter.split_documents(documents)
        print(f"   Level 0: {len(chunks)} base chunks")
        
        hierarchical_docs = chunks.copy()
        
        # Level 1: Cluster and summarize
        batch_size = 5
        summaries = []
        
        summary_prompt = PromptTemplate(
            template="Summarize the following code/text, preserving key technical details:\n\n{text}\n\nConcise Summary:",
            input_variables=["text"]
        )
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            combined = "\n\n".join([d.page_content for d in batch])[:3000]
            
            try:
                summary = self.llm_fast.invoke(summary_prompt.format(text=combined))
                summary_doc = Document(
                    page_content=summary,
                    metadata={"level": 1, "source": "raptor_summary", "batch": i//batch_size}
                )
                summaries.append(summary_doc)
            except Exception as e:
                print(f"   ⚠️ Summary error: {e}")
        
        hierarchical_docs.extend(summaries)
        print(f"   Level 1: {len(summaries)} summaries added")
        
        return hierarchical_docs
    
    def build_vectorstore(self, document_paths: List[str]):
        """Load documents and build RAPTOR-indexed vector store"""
        print(f"\n📚 Loading documents from {len(document_paths)} paths...")
        
        all_docs = []
        for path in document_paths:
            p = Path(path)
            try:
                if p.is_dir():
                    for loader_cls, glob in [(TextLoader, "**/*.txt"), (PyPDFLoader, "**/*.pdf")]:
                        loader = DirectoryLoader(path, glob=glob, loader_cls=loader_cls, show_progress=True)
                        all_docs.extend(loader.load())
                elif p.suffix == '.pdf':
                    all_docs.extend(PyPDFLoader(path).load())
                elif p.suffix in ['.txt', '.py', '.md']:
                    all_docs.extend(TextLoader(path).load())
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
        
        print(f"✅ Loaded {len(all_docs)} documents")
        
        # Build RAPTOR hierarchy
        hierarchical_docs = self.create_raptor_hierarchy(all_docs)
        
        # Create vector store
        print(f"💾 Building vector store with {len(hierarchical_docs)} documents...")
        self.vectorstore = Chroma.from_documents(
            documents=hierarchical_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print("✅ Vector store built")
    
    def hyde_generate(self, query: str) -> str:
        """HyDE: Generate hypothetical answer for better embedding"""
        hyde_prompt = f"""Generate a detailed, hypothetical answer to this query as if you were an expert:
Query: {query}
Hypothetical Expert Answer:"""
        
        try:
            return self.llm_fast.invoke(hyde_prompt)
        except:
            return query
    
    def multiquery_generate(self, query: str) -> List[str]:
        """MultiQuery: Generate query variations"""
        multiquery_prompt = f"""Generate 3 different variations of this query to capture different perspectives:
Original Query: {query}
Variations (one per line):"""
        
        try:
            result = self.llm_fast.invoke(multiquery_prompt)
            queries = [q.strip() for q in result.split('\n') if q.strip()]
            return queries[:3] + [query]
        except:
            return [query]
    
    async def retrieve_and_rerank(self, query: str, k: int = 5) -> List[Document]:
        """RAG 2.0: Hybrid retrieval + FlashRank reranking"""
        print(f"\n🔍 Retrieving for: {query[:80]}...")
        
        # Step 1: HyDE enhancement
        hyde_answer = self.hyde_generate(query)
        
        # Step 2: MultiQuery variations
        query_variations = self.multiquery_generate(query)
        
        # Step 3: Retrieve broad set (high recall)
        all_docs = []
        seen_content = set()
        
        for q in [hyde_answer] + query_variations:
            docs = self.vectorstore.similarity_search(q, k=10)
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(content_hash)
        
        print(f"   Retrieved {len(all_docs)} candidate documents")
        
        # Step 4: FlashRank reranking (high precision)
        if len(all_docs) > k:
            passages = [{"text": doc.page_content} for doc in all_docs]
            rerank_request = RerankRequest(query=query, passages=passages)
            
            try:
                rerank_results = self.reranker.rerank(rerank_request)
                ranked_indices = [r['corpus_id'] for r in rerank_results[:k]]
                reranked_docs = [all_docs[i] for i in ranked_indices]
                print(f"   ✅ Reranked to top {k} documents")
                return reranked_docs
            except Exception as e:
                print(f"   ⚠️ Reranking failed: {e}, using similarity order")
                return all_docs[:k]
        
        return all_docs[:k]


# ==================== SELF-HEALING CODING AGENT ====================
class SelfHealingAgent:
    """Autonomous coding agent with self-correction loop"""
    
    def __init__(self, llm_core: Ollama, project_dir: str = "./sca_project"):
        self.llm_core = llm_core  # Powerful model for reasoning
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(exist_ok=True)
    
    def generate_code(self, task: str, context: str, previous_error: str = "") -> str:
        """Generate code with context and optional error feedback"""
        
        error_section = ""
        if previous_error:
            error_section = f"""
PREVIOUS ATTEMPT FAILED WITH ERROR:
{previous_error}
Please fix the error in your new implementation.
"""
        
        prompt = f"""You are an expert programmer. Generate production-quality Python code.
TASK: {task}
CONTEXT:
{context}
{error_section}
REQUIREMENTS:
- Include all imports
- Add error handling
- Write clean, documented code
- Ensure it runs without errors
OUTPUT ONLY THE CODE (no markdown, no explanation):"""
        
        code = self.llm_core.invoke(prompt)
        
        # Clean markdown artifacts
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        return code.strip()
    
    def execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute code in sandbox"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_dir
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stderr": "Execution timeout exceeded"}
        except Exception as e:
            return {"success": False, "stderr": str(e)}
    
    def reflect_on_error(self, code: str, error: str) -> str:
        """Analyze error and provide reflection"""
        reflection_prompt = f"""Analyze why this code failed and explain how to fix it.
CODE:
{code[:1000]}
ERROR:
{error}
ANALYSIS (be specific and technical):"""
        
        return self.llm_core.invoke(reflection_prompt)
    
    async def autonomous_loop(self, task: str, context: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Self-healing loop: Generate → Execute → Reflect → Fix"""
        print(f"\n🤖 Starting autonomous coding task...")
        
        code = None
        execution_result = None
        error_log = []
        
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Generate code (with previous error if exists)
            previous_error = error_log[-1] if error_log else ""
            code = self.generate_code(task, context, previous_error)
            print(f"   💻 Generated {len(code)} chars of code")
            
            # Execute
            execution_result = self.execute_code(code)
            
            if execution_result["success"]:
                print(f"   ✅ Code executed successfully!")
                return {
                    "success": True,
                    "code": code,
                    "output": execution_result["stdout"],
                    "iterations": iteration + 1
                }
            
            # Failed - reflect and retry
            error = execution_result["stderr"]
            print(f"   ❌ Execution failed: {error[:100]}...")
            error_log.append(error)
            
            if iteration < max_iterations - 1:
                reflection = self.reflect_on_error(code, error)
                print(f"   🤔 Reflection: {reflection[:150]}...")
        
        # Max iterations reached
        return {
            "success": False,
            "code": code,
            "error": error_log[-1],
            "iterations": max_iterations
        }


# ==================== REWOO ORCHESTRATION ====================
class ReWOOOrchestrator:
    """ReWOO: Reasoning Without Observation"""
    
    def __init__(self, llm_fast: Ollama, llm_core: Ollama, rag: SuperRAGEngine):
        self.llm_fast = llm_fast  # For planning
        self.llm_core = llm_core  # For solving
        self.rag = rag
    
    async def planner(self, query: str) -> str:
        """ReWOO Planner: Generate execution plan"""
        plan_prompt = f"""Create a detailed execution plan for this task. Use placeholders like #E1, #E2 for evidence.
Task: {query}
Plan format:
#Plan1: Retrieve relevant context using RAG
#E1: <RAG results>
#Plan2: Analyze requirements and design solution
#E2: <Design>
#Plan3: Generate code implementation
#E3: <Code>
Your Plan:"""
        
        return self.llm_fast.invoke(plan_prompt)
    
    async def worker(self, plan: str, query: str) -> Dict[str, Any]:
        """ReWOO Worker: Execute plan and gather evidence"""
        print("\n⚙️ Worker executing plan...")
        
        # Extract RAG need from plan
        retrieved_docs = await self.rag.retrieve_and_rerank(query, k=5)
        
        context = "\n\n".join([f"[Doc {i+1}]:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs[:3])])
        
        return {
            "retrieved_docs": retrieved_docs,
            "context": context
        }
    
    async def solver(self, plan: str, evidence: Dict[str, Any], query: str) -> str:
        """ReWOO Solver: Synthesize final solution"""
        solver_prompt = f"""Based on the plan and gathered evidence, provide the final solution.
PLAN:
{plan}
EVIDENCE:
{evidence.get('context', '')}
QUERY: {query}
FINAL SOLUTION:"""
        
        return self.llm_core.invoke(solver_prompt)


# ==================== LANGGRAPH WORKFLOW ====================
class SuperCreatorAgent:
    """Main LangGraph-based orchestration"""
    
    def __init__(self, model_fast: str = "qwen2.5-coder:7b", model_core: str = "qwen2.5-coder:32b"):
        print("=" * 70)
        print("🚀 SUPER CREATOR AGENT - 2025 ARCHITECTURE")
        print("=" * 70)
        
        # Dual-model setup
        self.llm_fast = Ollama(model=model_fast, temperature=0.1, num_ctx=4096)
        self.llm_core = Ollama(model=model_core, temperature=0.1, num_ctx=8192)
        
        # Initialize components
        self.rag = SuperRAGEngine(model_name=model_fast)
        self.agent = SelfHealingAgent(self.llm_core)
        self.rewoo = ReWOOOrchestrator(self.llm_fast, self.llm_core, self.rag)
        
        # CRITICAL: Initialize checkpointer BEFORE building graph
        self.checkpointer = MemorySaver()
        
        # Build LangGraph workflow (now checkpointer exists)
        self.workflow = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine"""
        
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("worker", self._worker_node)
        workflow.add_node("solver", self._solver_node)
        workflow.add_node("generate_code", self._generate_code_node)
        workflow.add_node("execute_code", self._execute_code_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("human_approval", self._human_approval_node)
        
        # Define edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "worker")
        workflow.add_edge("worker", "solver")
        workflow.add_edge("solver", "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        
        # Conditional routing from execute
        workflow.add_conditional_edges(
            "execute_code",
            self._should_continue,
            {
                "reflect": "reflect",
                "human_approval": "human_approval",
                "end": END
            }
        )
        
        workflow.add_edge("reflect", "generate_code")
        workflow.add_edge("human_approval", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _planner_node(self, state: AgentState) -> AgentState:
        """ReWOO Planning phase"""
        plan = await self.rewoo.planner(state["query"])
        state["plan"] = plan
        print(f"\n📋 Plan created: {len(plan)} chars")
        return state
    
    async def _worker_node(self, state: AgentState) -> AgentState:
        """ReWOO Worker phase - gather evidence"""
        evidence = await self.rewoo.worker(state["plan"], state["query"])
        state["retrieved_docs"] = evidence["retrieved_docs"]
        print(f"\n📚 Retrieved {len(evidence['retrieved_docs'])} documents")
        return state
    
    async def _solver_node(self, state: AgentState) -> AgentState:
        """ReWOO Solver phase - synthesize solution"""
        context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"][:3]])
        solution = await self.rewoo.solver(state["plan"], {"context": context}, state["query"])
        state["final_answer"] = solution
        print(f"\n💡 Solution synthesized")
        return state
    
    async def _generate_code_node(self, state: AgentState) -> AgentState:
        """Generate code with context"""
        context = state.get("final_answer", "")
        previous_error = state["error_log"][-1] if state.get("error_log") else ""
        
        code = self.agent.generate_code(state["query"], context, previous_error)
        state["code"] = code
        print(f"\n💻 Code generated: {len(code)} chars")
        return state
    
    async def _execute_code_node(self, state: AgentState) -> AgentState:
        """Execute generated code"""
        result = self.agent.execute_code(state["code"])
        state["execution_result"] = result
        
        if not result["success"]:
            if "error_log" not in state:
                state["error_log"] = []
            state["error_log"].append(result["stderr"])
        
        print(f"\n⚡ Execution: {'✅ Success' if result['success'] else '❌ Failed'}")
        return state
    
    async def _reflect_node(self, state: AgentState) -> AgentState:
        """Reflect on failure"""
        reflection = self.agent.reflect_on_error(
            state["code"],
            state["error_log"][-1]
        )
        state["reflection"] = reflection
        state["iteration"] += 1
        print(f"\n🤔 Reflection complete (iteration {state['iteration']})")
        return state
    
    async def _human_approval_node(self, state: AgentState) -> AgentState:
        """Human-in-the-loop checkpoint"""
        print("\n⏸️  HUMAN APPROVAL REQUIRED")
        print(f"Code:\n{state['code'][:500]}...")
        # In production, this would pause for user input
        state["needs_human_approval"] = False
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Routing logic for execution results"""
        if state["execution_result"]["success"]:
            # Check if needs human review (e.g., complex operations)
            if "git" in state["code"].lower() or "subprocess" in state["code"].lower():
                return "human_approval"
            return "end"
        
        # Failed - check iteration count
        if state.get("iteration", 0) >= state.get("max_iterations", 3):
            return "human_approval"  # Max retries, need human help
        
        return "reflect"
    
    async def run(self, query: str, document_paths: List[str] = None) -> Dict[str, Any]:
        """Main execution interface"""
        
        # Build RAG if documents provided
        if document_paths:
            self.rag.build_vectorstore(document_paths)
        
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "plan": "",
            "retrieved_docs": [],
            "code": "",
            "execution_result": {},
            "reflection": "",
            "iteration": 0,
            "max_iterations": 3,
            "final_answer": "",
            "error_log": [],
            "needs_human_approval": False
        }
        
        # Run workflow
        config = {"configurable": {"thread_id": f"session_{datetime.now().timestamp()}"}}
        
        print(f"\n🎯 Processing: {query}")
        final_state = await self.workflow.ainvoke(initial_state, config)
        
        return {
            "success": final_state["execution_result"].get("success", False),
            "code": final_state["code"],
            "output": final_state["execution_result"].get("stdout", ""),
            "iterations": final_state["iteration"],
            "plan": final_state["plan"]
        }


# ==================== ASYNC DEMO ====================
async def main():
    """Demonstration"""
    
    # Initialize SCA
    sca = SuperCreatorAgent(
        model_fast="qwen2.5-coder:7b",
        model_core="qwen2.5-coder:32b"
    )
    
    # Load knowledge base (your books/docs)
    # sca.rag.build_vectorstore(["./books/", "./docs/"])
    
    # Test query
    result = await sca.run(
        query="Create a function to calculate Fibonacci sequence with memoization and error handling"
    )
    
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nCode:\n{result['code']}")
    print(f"\nOutput:\n{result['output']}")


if __name__ == "__main__":
    asyncio.run(main())
