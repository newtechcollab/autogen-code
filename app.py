# Flask App that wraps around AI SWE Agent
import logging
import requests
from flask import make_response, Flask, request, jsonify

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.tools import FunctionTool
from dotenv import load_dotenv
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import ChatMessage
from typing import Callable, Sequence
from autogen_core import CancellationToken
from autogen_agentchat.base import Response
from autogen_agentchat.base import TaskResult
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

import certifi
import urllib
import os 
import asyncio
import subprocess
import shlex

load_dotenv()

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where() 
os.environ["SSL_CERT_FILE"] = certifi.where()


app = Flask(__name__)

logging.basicConfig(filename='aiagent.log', level=logging.WARNING)

# Validate request
async def validate_request(request):

    # Get the secret
    secret = request.args.get('goponsobdo')

    if secret is None:
        raise Exception("Secret not provided")

    if secret != "amijantechai":
        raise Exception("Invalid secret")

# Handle GET request
@app.route("/", methods=["GET"])
async def handle_get():

    app.logger.warn("GET called")
    try:
        await validate_request(request)
    except Exception as e:
        app.logger.error(f"Exception is -> {e}")
        return "You are not authorized !!"

    # Identify the user, using the GitHub API token provided in the request headers.
    github_token = request.headers.get("X-GitHub-Token")

    # Get the User Query
    user_query = request.args.get('query')

    if user_query is None:
        return "You have not asked anything !!"

    # Process Request   
    return await process_request(request, github_token, user_query)

# Handle POST request
@app.route("/", methods=["POST"])
async def handle_post():

    app.logger.warn("POST called")

    # Identify the user, using the GitHub API token provided in the request headers.
    github_token = request.headers.get("X-GitHub-Token")

    # Parse the request payload and log it.
    payload = request.json

    if payload is None:
        app.logger.warn("Request Payload is empty, so returning")
        return "Invalid Request !!"
    else:
        app.logger.warn(f"Payload -> {payload}")

    user_query = payload.get('query')
    app.logger.warn(f"User Query -> {user_query}")
    if user_query is None:
        return "You have not asked anything !!"

    # Process Request   
    return process_request(request, github_token, user_query)

# Process request (GET or POST)
async def process_request(request, github_token, user_query) -> str:

    if github_token is None:
        app.logger.warn("GitHub Token not provided")
    else:
        app.logger.warn(f"Token -> {github_token}")
        user_response = requests.get("https://api.github.com/user", headers={"Authorization": f"token {github_token}"})
        user = user_response.json()
        app.logger.warn(f"User -> {user}")

    # Create an OpenAI-compatible model client.
    model_client = OpenAIChatCompletionClient(
        #base_url="https://api.githubcopilot.com/",
        #model="gpt-4o-2024-08-06",
        #api_key=token_for_user,
        model="gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Github MCP Server details
    github_server_params = StdioServerParams(
        command="docker",
        read_timeout_seconds=60,
        args=[
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "-e",
            "GH_HOST",
            "ghcr.io/github/github-mcp-server"
        ],
        env={
            "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
            "GH_HOST": os.getenv("GH_HOST")
        }
    )

    # JIRA MCP Server details
    jira_server_params = StdioServerParams(
        command="uv",
        read_timeout_seconds=60,
        args=[
            'run',
            'mcp-atlassian',
            '-v',
            '--jira-url',
            os.getenv('JIRA-URL'),
            '--jira-token',
            os.getenv('JIRA_TOKEN'),
            '--jira-username',
            os.getenv('JIRA-USERNAME'),
        ],
    )    

    # Define Github Tool
    github_tool = await mcp_server_tools(github_server_params)

    # Define JIRA Tool
    jira_tool = await mcp_server_tools(jira_server_params)

    # Create the JIRA agent
    jira_agent = AssistantAgent(
        "JiraAgent",
        description="An agent that performs various operations on JIRA such as getting Issue details",
        model_client=model_client,
        tools=jira_tool,
        system_message="""
        You are a helpful AI assistant that provides various operations on JIRA. 
        Use your tool to perform operations on JIRA such as getting details about an Issue 
        or getting details about a project etc. 
        Prepare a nice summary of the output.
        If you did not understand the query, simply return empty string.
        """,
    )

    # Create the Github agent
    github_agent = AssistantAgent(
        "GitHubAgent",
        description="An agent that performs various operations on Github such as searching for code in Github",
        model_client=model_client,
        tools=github_tool,
        system_message="""
        You are a helpful AI assistant that provides various operations on GitHub. 
        Use your tool to perform operations on Github such as cloning a repository or pull a request etc 
        Prepare a nice summary of the output.
        If you did not understand the query, simply return empty string.
        """,
    )

    # Command executor tool
    async def run_bash_command(cmd: str) -> str:
        """ Execute bash commands on local file system """
        app.logger.warn(f"Received command -> {cmd}")
        try:
            # shlex.split() safely splits the command string into a list of arguments
            command_args = shlex.split(cmd)
        
            result = subprocess.run(
                command_args, 
                capture_output=True, 
                text=True,
                check=True
            )
            app.logger.warn(f"Command output {result.stdout}") 
            return result.stdout
        
        except FileNotFoundError:
            app.logger.error(f"Command not found {command_args[0]}") 
            return f"Error: Command not found: {command_args[0]}"
        except subprocess.CalledProcessError as e:
            app.logger.error(f"Command error {e.stderr}") 
            return e.stderr
        except Exception as e:
            app.logger.error(f"Unknown error {e}") 
            return f"An unexpected error occurred: {e}"

    # Wrap command function as a Tool  
    command_executor_tool = FunctionTool(run_bash_command, description="Bash Command Executor", strict=False)

    # Create the BugAnalyzer agent
    buganalyzer_agent = AssistantAgent(
        "BugAnalyzerAgent",
        description="An agent that analyzes a given JIRA Issue",
        model_client=model_client,
        tools=[command_executor_tool],
        system_message="""
        You are a helpful AI assistant. 
        Use following steps   
        1. Clone the Github repository from https://github.com/newtechcollab/sample-ecommerce-python.git into /tmp directory. If the clone fails due to existing local directory in /tmp, then delete the directory in /tmp and then clone the repository. Issue separate commands for each operation. Do not issue a single command.
        2. Search for files in /tmp directory that includes phrases used in JIRA Issue 
        3. Identify the file names that come up in the search query 2 above.
        4. If no files are found or if any error occurs while searching for files, then return with a message "I am unable to identify which source code to be fixed"
        5. Read these files and identify the code changes that have to be made to fix the JIRA Issue
        Prepare a nice HTML-formatted report, describing the Issue and the recommended code fix
        Start the report with the Title as Issue details and Recommended Code fix
        Then, print your recommended fix with heading "Here is my analysis"  
        Print the names of files that were identified in step 3 above.
        Print the code changes that have to be made
        End with the text "COMPLETED"
        """,
    )

    orchestrator_agent = AssistantAgent(
        "OrchestratorAgent",
        description="An agent for orchestrating tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            JiraAgent: Performs operations on JIRA such as retrieving Issue details
            BugAnalyzerAgent: Analyzes a given JIRA Issue and recommends code fixes

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        Print the response generated by every agent after every subtask is coompleted.
        After all tasks are complete, summarize the findings and end with "COMPLETED".
        """,
    )

    # Define a termination condition that stops the task if the BugFixer returns with 'COMPLETED'.
    text_termination = TextMentionTermination("COMPLETED")
    max_msg_termination = MaxMessageTermination(max_messages=20)
    termination = text_termination | max_msg_termination
    #termination = text_termination 

    #Create a team with the JIRA downloader and BugFixer agents.
    team = RoundRobinGroupChat([jira_agent, buganalyzer_agent], termination_condition=text_termination)
    #team = RoundRobinGroupChat([jira_agent, github_agent, bugfixer_agent], termination_condition=max_msg_termination)
    """
    team = SelectorGroupChat(
        [orchestrator_agent, jira_agent, buganalyzer_agent], 
        termination_condition=termination,
        model_client=model_client,
        allow_repeated_speaker=True,
    )
    """

    result = await team.run(task=user_query)
    #result = await buganalyzer_agent.run(task=user_query)
    app.logger.warn("FINAL RESPONSE")
    lines=[]
    for message in reversed(result.messages):
        if isinstance(message.content, str):
            line=message.content
            lines.append(line)
            lines.append("\n\n\n")
            app.logger.warn(f"  Source: {message.source}, Content: {message.content}")
            break
    return " ".join(lines)

    """
    app.logger.warn(result.content)
    response = make_response(jsonify(result.conent), 200)
    response.headers['Content-Type'] = 'application/json'
    return response
    """

    """
    def stream_autogen_response():
        await message in team.run_stream(task="Fix JIRA defects"):
            app.logger.warn("Response from Team")
            app.logger.warn(message)
            yield message

    return app.response_class(stream_autogen_response(), mimetype='text/event-stream')

    # Use Copilot's LLM to generate a response to the user's messages, with
    # our extra system messages attached.
    copilot_response = requests.post(
        "https://api.githubcopilot.com/chat/completions",
        headers={
            "Authorization": f"Bearer {token_for_user}",
            "Content-Type": "application/json"
        },
        json={
            "messages": messages,
            "stream": True
        },
        stream=True
    )

    # Stream the response straight back to the user.
    app.logger.warn("Returning response")
    app.logger.warn(messages[-1])
    #return app.response_class(messages[-1], mimetype='application/text')

    output = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "model": "gpt-4-1106-preview",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": messages[-1],
                },
            }
        ]
    }
    response = make_response(jsonify(output), 200)
    response.headers['Content-Type'] = 'application/json'
    return response
    #return jsonify(output)
    #return app.response_class(jsonify(output), mimetype='application/json')
    """
