<div id="top">

<!-- HEADER STYLE: COMPACT -->
<img src="multi-language-model-evaluation-framework.png" width="30%" align="left" style="margin-right: 15px">

# MULTI-LANGUAGE-MODEL EVALUATION FRAMEWORK
<em>Master Every Model. Compare. Improve. Succeed Faster.</em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat-square&logo=Streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/TOML-9C4121.svg?style=flat-square&logo=TOML&logoColor=white" alt="TOML">
<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=flat-square&logo=FastAPI&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/LangChain-1C3C3C.svg?style=flat-square&logo=LangChain&logoColor=white" alt="LangChain">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=flat-square&logo=OpenAI&logoColor=white" alt="OpenAI">
<img src="https://img.shields.io/badge/uv-DE5FE9.svg?style=flat-square&logo=uv&logoColor=white" alt="uv">
<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=flat-square&logo=Pydantic&logoColor=white" alt="Pydantic">

<br clear="left"/>

## ☀️ Table of Contents

- [☀ ️ Table of Contents](#-table-of-contents)
- [🌞 Overview](#-overview)
- [🔥 Features](#-features)
- [🌅 Project Structure](#-project-structure)
    - [🌄 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
    - [🌟 Prerequisites](#-prerequisites)
    - [⚡ Installation](#-installation)
    - [🔆 Usage](#-usage)
    - [🌠 Testing](#-testing)
- [🌻 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [✨ Acknowledgments](#-acknowledgments)

---

## 🌞 Overview

A lightweight multi-language-model evaluation framework powered by LiteLLM, Langfuse, and Ragas. Run and compare multiple LLMs through a unified API, with full tracing, metrics, and quality scoring. Ideal for benchmarking language models, and rapid model experimentation.

**Why multi-language-model-evaluation-framework?**

This project empowers you to efficiently benchmark language models with advanced configuration, extensibility, and robust integrations. The core features include:

- **📦 Centralized Environment Management:** Simplifies installation and maintains compatibility with essential libraries.
- **📊 Multi-metric Evaluation:** Supports comprehensive analysis across multiple evaluation metrics.
- **🔗 Seamless API Integrations:** Effortlessly connect to external APIs for model or data ingestion.
- **🛠️ Modular Architecture:** Easily extend or adapt components to fit your workflow.
- **🌐 Robust Data Handling:** Efficiently manage diverse datasets for consistent and scalable evaluations.

---

## 🔥 Features

|      | Component       | Details                              |
| :--- | :-------------- | :----------------------------------- |
| ⚙️  | **Architecture**  | <ul><li>Python-based backend</li><li>Modular evaluation pipeline</li><li>Streamlit & FastAPI for UI/API</li><li>Orchestrates multi-LLM evaluation</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Leverages Pydantic for types</li><li>Modern pyproject.toml management</li><li>Uses LangChain abstraction</li></ul> |
| 📄 | **Documentation** | <ul><li>No dedicated docs directory</li><li>Standard Python docstrings (inferred)</li><li>Dependency-based documentation (pyproject.toml)</li></ul> |
| 🔌 | **Integrations**  | <ul><li>OpenAI, LangChain, LiteLLM</li><li>OpenTelemetry for tracing</li><li>Export to Excel (.xlsxwriter, openpyxl)</li><li>Langfuse, Langgraph</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Composable Langchain modules</li><li>Separate backend/frontend (FastAPI/Streamlit)</li><li>Flexible evaluation components</li></ul> |
| 🧪 | **Testing**       | <ul><li>No evidence of test framework</li><li>Manual/exploratory via UI</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Async-friendly frameworks (FastAPI, Uvicorn)</li><li>Optimized Excel I/O</li></ul> |
| 🛡️ | **Security**      | <ul><li>python-dotenv for secrets</li><li>Cloud API integrations</li><li>Best-practice dependency use</li></ul> |
| 📦 | **Dependencies**  | <ul><li>openpyxl, xlsxwriter</li><li>networkx, matplotlib</li><li>playwright for browser automation</li><li>fastapi, streamlit, uvicorn</li><li>pydantic, pandas</li><li>openai, langchain-core</li></ul> |

---

## 🌅 Project Structure

```sh
└── multi-language-model-evaluation-framework/
    ├── multi-language-model-evaluation-framework.png
    ├── pyproject.toml
    └── uv.lock
```

### 🌄 Project Index

<details open>
	<summary><b><code>MULTI-LANGUAGE-MODEL-EVALUATION-FRAMEWORK</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\l.inghilterra\Desktop\Lavoro\multi-language-model-evaluation-framework/blob/master/pyproject.toml'>pyproject.toml</a></b></td>
					<td style='padding: 8px;'>- Project metadata and dependency management are defined to ensure smooth installation, compatibility, and operation of the multi-language model evaluation framework<br>- By specifying essential libraries and configuration details, this central manifest orchestrates the environment required for evaluating language models across datasets, supporting features such as multi-metric analysis, API integrations, and robust data handling across the project’s modular architecture.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## 🚀 Getting Started

### 🌟 Prerequisites

This project requires the following dependencies:

- **Programming Language:** unknown
- **Package Manager:** Uv

### ⚡ Installation

Build multi-language-model-evaluation-framework from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone ../multi-language-model-evaluation-framework
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd multi-language-model-evaluation-framework
    ```

3. **Install the dependencies:**

<!-- SHIELDS BADGE CURRENTLY DISABLED -->
	<!-- [![uv][uv-shield]][uv-link] -->
	<!-- REFERENCE LINKS -->
	<!-- [uv-shield]: None -->
	<!-- [uv-link]: None -->

	**Using [uv](None):**

	```sh
	❯ echo 'INSERT-INSTALL-COMMAND-HERE'
	```

### 🔆 Usage

Run the project with:

**Using [uv](None):**
```sh
echo 'INSERT-RUN-COMMAND-HERE'
```

### 🌠 Testing

Multi-language-model-evaluation-framework uses the {__test_framework__} test framework. Run the test suite with:

**Using [uv](None):**
```sh
echo 'INSERT-TEST-COMMAND-HERE'
```

---

## 🌻 Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## 🤝 Contributing

- **💬 [Join the Discussions](https://LOCAL/Lavoro/multi-language-model-evaluation-framework/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://LOCAL/Lavoro/multi-language-model-evaluation-framework/issues)**: Submit bugs found or log feature requests for the `multi-language-model-evaluation-framework` project.
- **💡 [Submit Pull Requests](https://LOCAL/Lavoro/multi-language-model-evaluation-framework/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your LOCAL account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone C:\Users\l.inghilterra\Desktop\Lavoro\multi-language-model-evaluation-framework
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to LOCAL**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://LOCAL{/Lavoro/multi-language-model-evaluation-framework/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Lavoro/multi-language-model-evaluation-framework">
   </a>
</p>
</details>

---

## 📜 License

Multi-language-model-evaluation-framework is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## ✨ Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
