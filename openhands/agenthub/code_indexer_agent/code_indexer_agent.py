import fnmatch
import os
import re
from typing import Any, Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from tree_sitter import Parser
from tree_sitter_languages import get_language, get_parser

from openhands.controller.agent import Agent
from openhands.events.action import CodeResponseAction, ErrorAction, NoOpAction
from openhands.events.action.action import Action
from openhands.events.observation import (
    CodeQueryObservation,
    CommandOutputObservation,
    ErrorObservation,
    FileContentObservation,
)


class CodeIndexerAgent(Agent):
    """Agent for indexing and retrieving code documentation using RAG."""

    # Common file patterns to index across different project types
    FILE_PATTERNS = {
        # Source code files
        '*.py',
        '*.java',
        '*.cpp',
        '*.c',
        '*.h',
        '*.hpp',
        '*.cs',
        '*.js',
        '*.ts',
        '*.go',
        '*.rs',
        '*.rb',
        '*.php',
        '*.scala',
        '*.kt',
        '*.swift',
        '*.m',
        # Web files
        '*.html',
        '*.css',
        '*.jsx',
        '*.tsx',
        '*.vue',
        '*.svelte',
        # Config files
        '*.json',
        '*.yaml',
        '*.yml',
        '*.toml',
        '*.ini',
        '*.conf',
        # Documentation
        '*.md',
        '*.rst',
        '*.txt',
        '*.pdf',
        # Template files
        '*.j2',
        '*.jinja',
        '*.template',
        '*.tpl',
        # Shell scripts
        '*.sh',
        '*.bash',
        '*.zsh',
        '*.fish',
        # Build files
        'Makefile',
        '*.mk',
        'CMakeLists.txt',
        '*.gradle',
        '*.maven',
        # Docker files
        'Dockerfile*',
        'docker-compose*.yml',
        # Other important files
        '.env.example',
        'requirements.txt',
        'package.json',
        'cargo.toml',
    }

    # Common patterns to exclude based on .gitignore conventions
    EXCLUDE_PATTERNS = {
        # Build outputs and caches
        '**/build/**',
        '**/dist/**',
        '**/out/**',
        '**/target/**',
        '**/.cache/**',
        '**/__pycache__/**',
        '**/*.pyc',
        '**/*.pyo',
        '**/.pytest_cache/**',
        '**/.coverage/**',
        '**/htmlcov/**',
        # Dependencies
        '**/node_modules/**',
        '**/venv/**',
        '**/.env/**',
        '**/.venv/**',
        '**/vendor/**',
        '**/packages/**',
        '**/.bundle/**',
        # IDE and editor files
        '**/.idea/**',
        '**/.vscode/**',
        '**/.vs/**',
        '**/*.swp',
        '**/*.swo',
        '**/*~',
        '**/.DS_Store',
        # VCS directories
        '**/.git/**',
        '**/.svn/**',
        '**/.hg/**',
        # Logs and temporary files
        '**/logs/**',
        '**/log/**',
        '**/tmp/**',
        '**/temp/**',
        '**/*.log',
        '**/*.tmp',
        # Generated documentation
        '**/docs/_build/**',
        '**/site/**',
        # Binary and media files
        '**/*.exe',
        '**/*.dll',
        '**/*.so',
        '**/*.dylib',
        '**/*.zip',
        '**/*.tar',
        '**/*.gz',
        '**/*.rar',
        '**/*.jpg',
        '**/*.jpeg',
        '**/*.png',
        '**/*.gif',
        '**/*.ico',
        '**/*.mp3',
        '**/*.mp4',
        '**/*.wav',
        '**/*.avi',
        # Database files
        '**/*.db',
        '**/*.sqlite',
        '**/*.sqlite3',
        # Secrets and credentials
        '**/.env',
        '**/*.pem',
        '**/*.key',
        '**/*.cert',
        '**/secrets/**',
        '**/credentials/**',
    }

    def __init__(self, config: Dict[str, Any], codebase_path: str = '.', **kwargs):
        """Initialize the agent with config and optional kwargs."""
        super().__init__(
            config=config, llm=kwargs.get('llm', ChatOpenAI(temperature=0))
        )
        self.codebase_path = codebase_path
        self.embeddings = kwargs.get('embeddings', OpenAIEmbeddings())
        self.llm = kwargs.get('llm', ChatOpenAI(temperature=0))
        self.vector_store = None
        self.parsers = self._initialize_parsers()
        self.initialize_rag()

    def _initialize_parsers(self) -> Dict[str, Parser]:
        """Initialize tree-sitter parsers for supported languages."""
        parsers = {}
        try:
            parsers['.py'] = {
                'language': get_language('python'),
                'parser': get_parser('python'),
            }
            parsers['.js'] = {
                'language': get_language('javascript'),
                'parser': get_parser('javascript'),
            }
            parsers['.go'] = {
                'language': get_language('go'),
                'parser': get_parser('go'),
            }
        except Exception as e:
            print(f'Error initializing parsers: {e}')
        return parsers

    def _parse_code_structure(self, file_path: str, content: str) -> List[Document]:
        """Parse code structure using AST for Python files and basic parsing for others."""
        documents = []
        ext = os.path.splitext(file_path)[1]

        # Add the full file content as a document
        documents.append(
            Document(
                page_content=content,
                metadata={
                    'source': file_path,
                    'type': 'file',
                    'name': os.path.basename(file_path),
                },
            )
        )

        try:
            if ext == '.py':
                # Use AST for Python files
                import ast

                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Get the source lines for the class
                        class_lines = content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ]
                        class_source = '\n'.join(class_lines)

                        documents.append(
                            Document(
                                page_content=class_source,
                                metadata={
                                    'source': file_path,
                                    'type': 'class',
                                    'name': node.name,
                                    'start_line': node.lineno,
                                    'end_line': node.end_lineno,
                                },
                            )
                        )

                        # Extract methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method_lines = content.splitlines()[
                                    item.lineno - 1 : item.end_lineno
                                ]
                                method_source = '\n'.join(method_lines)
                                documents.append(
                                    Document(
                                        page_content=method_source,
                                        metadata={
                                            'source': file_path,
                                            'type': 'method',
                                            'name': item.name,
                                            'class': node.name,
                                            'start_line': item.lineno,
                                            'end_line': item.end_lineno,
                                        },
                                    )
                                )

                    if isinstance(node, ast.FunctionDef) and isinstance(
                        node.parent, ast.Module
                    ):
                        # Only process top-level functions
                        func_lines = content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ]
                        func_source = '\n'.join(func_lines)
                        documents.append(
                            Document(
                                page_content=func_source,
                                metadata={
                                    'source': file_path,
                                    'type': 'function',
                                    'name': node.name,
                                    'start_line': node.lineno,
                                    'end_line': node.end_lineno,
                                },
                            )
                        )

            elif ext in ['.md', '.rst', '.txt']:
                # Process documentation files
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            'source': file_path,
                            'type': 'documentation',
                            'name': os.path.basename(file_path),
                        },
                    )
                )

            else:
                # Basic parsing for other files
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            'source': file_path,
                            'type': 'text',
                            'name': os.path.basename(file_path),
                        },
                    )
                )

        except Exception as e:
            print(f'Error processing {file_path}: {e}')
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        'source': file_path,
                        'type': 'text',
                        'name': os.path.basename(file_path),
                    },
                )
            )

        return documents

    def _get_gitignore_patterns(self) -> set:
        """Read .gitignore files and return their patterns."""
        patterns = set()
        for root, _, files in os.walk('.'):
            if '.gitignore' in files:
                with open(os.path.join(root, '.gitignore')) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Convert .gitignore pattern to glob pattern
                            if not line.startswith('**/'):
                                line = f'**/{line}'
                            if not line.endswith('/**') and '.' not in line:
                                line = f'{line}/**'
                            patterns.add(line)
        return patterns

    def _convert_pattern_to_regex(self, pattern: str) -> re.Pattern:
        """Convert a glob pattern to a regex pattern."""
        # Escape special characters
        pattern = fnmatch.translate(pattern)
        # Convert the shell pattern to regex pattern
        return re.compile(pattern)

    def _matches_any_pattern(self, file_path: str, patterns: set) -> bool:
        """Check if file_path matches any of the exclude patterns."""
        for pattern in patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        return False

    def initialize_rag(self):
        """Initialize the RAG components including vector store and retrieval chain."""
        # Combine built-in exclude patterns with .gitignore patterns
        exclude_patterns = set(self.EXCLUDE_PATTERNS)
        exclude_patterns.update(self._get_gitignore_patterns())

        documents = []
        # Walk through the codebase and parse files
        for root, _, files in os.walk(self.codebase_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.codebase_path)
                ext = os.path.splitext(file)[1]

                # Skip excluded files using fnmatch
                if self._matches_any_pattern(rel_path, exclude_patterns):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if ext == '.py':
                        # Use our basic Python parser
                        parsed_docs = self._parse_code_structure(file_path, content)
                    else:
                        # For non-Python files, create a simple document
                        parsed_docs = [
                            Document(
                                page_content=content,
                                metadata={'source': file_path, 'type': 'text'},
                            )
                        ]
                    documents.extend(parsed_docs)
                except Exception as e:
                    print(f'Error processing {file_path}: {e}')

        # Check if we have any documents
        if not documents:
            print('Warning: No documents found to index!')
            # Create a dummy document to initialize the vector store
            documents = [
                Document(
                    page_content='Placeholder document',
                    metadata={'source': 'placeholder', 'type': 'text'},
                )
            ]

        # Split documents if they're too large
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=['\n\n', '\n', ' ', ''],
        )
        splits = text_splitter.split_documents(documents)

        # Ensure we have valid splits
        if not splits:
            raise ValueError('No valid documents to index after splitting!')

        # Create vector store using Chroma with filtered metadata

        # Filter complex metadata before creating the vector store
        filtered_splits = []
        for doc in splits:
            # Create a new metadata dictionary with only simple types
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                # Only keep simple types
                if isinstance(value, (str, bool, int, float)):
                    filtered_metadata[key] = value
                elif value is None:
                    filtered_metadata[key] = ''  # Convert None to empty string
                else:
                    # Convert complex types to string representation
                    filtered_metadata[key] = str(value)

            # Create new document with filtered metadata
            filtered_splits.append(
                Document(page_content=doc.page_content, metadata=filtered_metadata)
            )

        # Create vector store using filtered documents
        self.vector_store = Chroma.from_documents(
            documents=filtered_splits,
            embedding=self.embeddings,
            collection_name='code_index',
            persist_directory='.chromadb',
        )

        # Create RAG prompt with improved context understanding
        self.rag_prompt = ChatPromptTemplate.from_template("""
        You are a code documentation assistant. Use the following context to answer
        questions about the codebase. If you don't know the answer, say so.

        When answering:
        1. Prioritize the most relevant information
        2. Include code examples when appropriate
        3. Mention relationships between different code elements
        4. For questions about class methods, ALWAYS list ALL available methods including __init__
        5. For implementation details, explain the logic clearly
        6. When describing classes or functions, include their full signatures
        7. Always mention the file path where code elements are found

        Context: {context}

        Question: {question}

        Answer in a structured format:
        - Location: [file path]
        - Type: [class/function/etc]
        - Name: [element name]
        - Details: [your detailed response]
        """)

        # Create retrieval chain with improved search
        self.retrieval_chain = (
            {
                'context': self.vector_store.as_retriever(
                    search_type='mmr',
                    search_kwargs={'k': 8},  # Increased for better context
                ),
                'question': RunnablePassthrough(),
            }
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def search(self, query: str) -> List[Document]:
        """Search the codebase for relevant documents."""
        if not query:
            raise ValueError('Query cannot be empty')

        if not self.vector_store:
            raise RuntimeError('Vector store not initialized')

        try:
            # Use MMR search for better diversity in results
            results = self.vector_store.max_marginal_relevance_search(
                query,
                k=10,  # Increased number of results
                fetch_k=20,  # Fetch more candidates for better diversity
                lambda_mult=0.7,  # Balance between relevance and diversity
            )

            # Filter and sort results
            filtered_results = []
            for doc in results:
                # Skip empty or irrelevant documents
                if not doc.page_content.strip():
                    continue

                # Add relevance score to metadata
                doc.metadata['relevance'] = self._calculate_relevance(
                    query, doc.page_content
                )
                filtered_results.append(doc)

            # Sort by relevance
            filtered_results.sort(
                key=lambda x: x.metadata.get('relevance', 0), reverse=True
            )

            return filtered_results

        except Exception as e:
            print(f'Search error: {e}')
            return []

    def query(self, question: str) -> str:
        """Query the codebase using the RAG chain."""
        if not question:
            raise ValueError('Query cannot be empty')

        if not self.retrieval_chain:
            raise RuntimeError('Retrieval chain not initialized')

        try:
            # Get relevant documents
            docs = self.search(question)
            if not docs:
                return 'No relevant information found'

            # Format context from documents
            context = '\n\n'.join(
                f"[{doc.metadata.get('type', 'unknown')}] {doc.metadata.get('source', 'unknown')}"
                f"\n{doc.page_content}"
                for doc in docs
            )

            # Create prompt
            prompt = f"""
            Based on the following code context, please answer the question.

            Context:
            {context}

            Question: {question}

            Answer in this format:
            Location: [file path]
            Type: [class/function/etc]
            Name: [element name]
            Details: [detailed explanation]
            """

            # Get response from LLM
            try:
                # Try using invoke method first
                response = self.llm.invoke(prompt)
                if isinstance(response, str):
                    return response
                # If response is a ChatResult or similar, convert to string
                return str(response)
            except (AttributeError, TypeError):
                # Fallback to _generate if invoke is not available
                result = self.llm._generate([prompt])
                if result and result.generations:
                    return str(result.generations[0].message)
                return 'Error: Could not generate response'

        except Exception as e:
            print(f'Query error: {e}')
            return f"""
            Location: unknown
            Type: error
            Name: error
            Details: Error processing query: {str(e)}
            """

    def step(self, state) -> Action:
        """Process the current state and return next action."""
        observation = state.get_last_observation()

        if isinstance(observation, CodeQueryObservation):
            if not observation.query:
                return ErrorAction(error='Query cannot be empty')
            try:
                response = self.query(observation.query)
                if not response or not response.strip():
                    return ErrorAction(error='No relevant information found')

                # Always return formatted response
                if not all(
                    marker in response
                    for marker in ['Location:', 'Type:', 'Name:', 'Details:']
                ):
                    response = f"""
                    Location: unknown
                    Type: unknown
                    Name: unknown
                    Details: {response}
                    """

                return CodeResponseAction(response=response)

            except Exception as e:
                return ErrorAction(error=str(e))

        elif isinstance(
            observation, (CommandOutputObservation, FileContentObservation)
        ):
            return NoOpAction()

        elif isinstance(observation, ErrorObservation):
            return ErrorAction(error=observation.error)

        return NoOpAction()

    def update_index(self):
        """Update the vector store with any new documents."""
        exclude_patterns = set(self.EXCLUDE_PATTERNS)
        exclude_patterns.update(self._get_gitignore_patterns())

        documents = []
        # Walk through the codebase and parse files
        for root, _, files in os.walk(self.codebase_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.codebase_path)
                ext = os.path.splitext(file)[1]

                # Skip excluded files using fnmatch
                if self._matches_any_pattern(rel_path, exclude_patterns):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if ext == '.py':
                        # Use our basic Python parser
                        parsed_docs = self._parse_code_structure(file_path, content)
                    else:
                        # For non-Python files, create a simple document
                        parsed_docs = [
                            Document(
                                page_content=content,
                                metadata={'source': file_path, 'type': 'text'},
                            )
                        ]
                    documents.extend(parsed_docs)
                except Exception as e:
                    print(f'Error processing {file_path}: {e}')

        # Split documents if they're too large
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=['\n\n', '\n', ' ', ''],
        )
        splits = text_splitter.split_documents(documents)

        # Filter metadata before adding to vector store
        filtered_splits = []
        for doc in splits:
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, bool, int, float)):
                    filtered_metadata[key] = value
                elif value is None:
                    filtered_metadata[key] = ''
                else:
                    filtered_metadata[key] = str(value)

            filtered_splits.append(
                Document(page_content=doc.page_content, metadata=filtered_metadata)
            )

        # Create a new collection with updated documents
        collection_name = 'code_index'
        persist_directory = '.chromadb'

        # Create new vector store
        new_vector_store = Chroma.from_documents(
            documents=filtered_splits,
            embedding=self.embeddings,
            collection_name=f'{collection_name}_new',
            persist_directory=persist_directory,
        )

        # Replace old vector store with new one
        self.vector_store = new_vector_store

        # Update the retrieval chain to use the new vector store
        self.retrieval_chain = (
            {
                'context': self.vector_store.as_retriever(
                    search_type='mmr', search_kwargs={'k': 8}
                ),
                'question': RunnablePassthrough(),
            }
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content."""
        # Simple relevance calculation based on term frequency
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())

        # Calculate Jaccard similarity
        intersection = len(query_terms & content_terms)
        union = len(query_terms | content_terms)

        if union == 0:
            return 0

        return intersection / union
