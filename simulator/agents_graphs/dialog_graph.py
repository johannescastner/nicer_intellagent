from langchain_core.runnables.base import Runnable
from typing import Callable
from langgraph.graph import END
from typing import Annotated
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict
from typing import Optional
import time
from langgraph.graph.message import add_messages
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
from simulator.utils.llm_utils import convert_messages_to_str
import json


class DialogState(TypedDict):
    user_messages: Annotated[list, add_messages]
    chatbot_messages: Annotated[list, add_messages]
    chatbot_args: Optional[dict]
    thread_id: str
    user_thoughts: Optional[list]
    critique_feedback: Optional[str]
    stop_signal: Optional[str]


class Dialog:
    """
    Building the dialog graph that runs the convesration between the chatbot and the user
    """

    def __init__(self, user: Runnable, chatbot: Runnable, critique: Runnable, intermediate_processing: Callable = None,
                 memory=None):
        """
        Initialize the event generator.]
        :param user (Runnable): The user model
        :param chatbot (Runnable): The chatbot model
        :param intermediate_processing (optional): A function between that processes the output of the user and the
        chatbot at each step
        :param critique (Runnable): The critique mode, should determine if the final decision of the user is correct
        :param memory (optional): The memory to store the conversations artifacts
        """
        self.user = user
        self.chatbot = chatbot
        self.critique = critique
        self.intermediate_processing = intermediate_processing  # TODO: Add default function
        self.memory = memory
        self.compile_graph()

    def user_end_condition(self, state: DialogState):
        """Condition function for user node: goes to chatbot or end_critique"""
        # After user node, check if we should go to chatbot or end_critique
        if state['stop_signal'] != '':  # User sent ###STOP
            return 'end_critique'  # Get critique
        else:
            return 'chatbot'  # Continue conversation
    
    def critique_end_condition(self, state: DialogState):
        """Condition function for end_critique node: goes to user or END"""
        # After end_critique node, check if we should go to user or END
        if self.intermediate_processing(state) == 'END':
            return END
        else:
            return 'user'


    def simulated_user_node(self, state):
        messages = [state["user_messages"][0]] + set_user_message(state)
        response = self.user.invoke(messages)
        
        # Initialize user_thoughts
        user_thoughts = state['user_thoughts'] if state['user_thoughts'] is not None else []

        # Key: Separate "Add to State" and "Save to Database"
        #Step 1: If you have an idea, add it to the state (independent of memory)
        if response['thought'] is not None:
            user_thoughts.append(response['thought'])
        
        #Step 2: If memory is configured, save it to the database
        if self.memory is not None:
            if response['thought'] is not None:
                self.memory.insert_thought(state['thread_id'], response['thought'])
            self.memory.insert_dialog(state['thread_id'], 'Human', response['response'])
        
        result_state = {'user_thoughts': user_thoughts, 'critique_feedback': '', 'stop_signal': ''}
        if '###STOP' in response['response']:
            result_state['stop_signal'] = response['response']
        else:
            result_state.update({"chatbot_messages": [HumanMessage(content=response['response'])],
                                'user_messages': [AIMessage(content=response['response'])]})
        return result_state

    def critique_node(self, state):
        # Call the simulated user
        if 'Thought:' in state['user_thoughts'][-1]:
            user_thought = state['user_thoughts'][-1].split('Thought:')[1]
        else:
            user_thought = state['user_thoughts'][-1]
        conversation = convert_messages_to_str(state['chatbot_messages'], True)
        if '###STOP FAILURE' in state['chatbot_messages'][-1].content:
            judgement = f"The chatbot failed to adhere the policies\n Reason:{user_thought}"
        else:
            judgement = f"The chatbot adhered to the policies\n Reason:{user_thought}"
        response = self.critique.invoke({'reason': judgement, 'conversation': conversation})
        return {"critique_feedback": response.content}

    def chat_bot_node(self, state):
        messages = state["chatbot_messages"]
        args = dict(state.get('chatbot_args', None) or {})
        args['thread_id'] = state.get('thread_id', '')
        response = self.chatbot.invoke({'messages': messages, 'args': args})
        last_human_message = max([i for i, v in enumerate(response['messages']) if v.type == 'human'])
        all_tool_calls = {}
        if self.memory is not None:
            # Inserting tool calls into memory
            for message in response['messages'][last_human_message + 1:]:
                if hasattr(message, 'tool_calls'):
                    for tool_call in message.tool_calls:
                        all_tool_calls[tool_call['id']] = tool_call
                if message.type == 'tool':
                    all_tool_calls[message.tool_call_id]['output'] = message.content
            for v in all_tool_calls.values():
                self.memory.insert_tool(state['thread_id'], v['name'], json.dumps(v['args']), v['output'])
                time.sleep(0.001)
            # inserting the chatbot messages into memory
            self.memory.insert_dialog(state['thread_id'], 'AI', response['messages'][-1].content)
        return {"chatbot_messages": response['messages'][last_human_message+1:],
                'user_messages': [HumanMessage(content=response['messages'][-1].content)]}

    def compile_graph(self):
        workflow = StateGraph(DialogState)
        workflow.add_node("user", self.simulated_user_node)
        workflow.add_node("chatbot", self.chat_bot_node)
        workflow.add_node("end_critique", self.critique_node)
        workflow.add_edge(START, "user")
        workflow.add_conditional_edges(
            "user",
            self.user_end_condition,
            ["chatbot", "end_critique"],
        )
        workflow.add_conditional_edges(
            "end_critique",
            self.critique_end_condition,
            ["user", END],
        )
        workflow.add_edge("chatbot", "user")
        self.graph = workflow.compile()

    def invoke(self, **kwargs):
        """
        Invoke the agent with the messages
        :return:
        """
        return self.graph.invoke(**kwargs)

    def ainvoke(self, **kwargs):
        """
        async Invoke the agent with the messages
        :return:
        """
        return self.graph.ainvoke(**kwargs)


def set_user_message(state: DialogState) -> list[BaseMessage]:
    """
    Set the user message
    :param state: The current state
    :return: The AI message
    """
    conversation = convert_messages_to_str(state['chatbot_messages'])
    text = f"You are provided with the conversation between the user and the chatbot.\n# Conversation:\n{conversation}"
    messages_list = [HumanMessage(content=text)]
    critique_feedback = state.get('critique_feedback', '')
    if not critique_feedback == '':
        text = f"{state['user_thoughts'][-1]}\nUser Response:\n{state['stop_signal']}"
        messages_list.append(AIMessage(content=text))
        text = 'Your response was inaccurate, you are provided with the feedback from the critique. ' \
               f'Please provide a new response (use the same format), you should also determine if the conversation should continue or stop.\nFeedback:\n{state["critique_feedback"]}'
        messages_list.append(HumanMessage(content=text))
    return messages_list
