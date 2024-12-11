"""
An example agent for Bank Support.
"""
import argparse
import asyncio
from dataclasses import dataclass

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from utils.llm_model import LLMModel

logfire.configure()


class DatabaseConn:
    """
    This is a fake database class. In reality, it would connect to the database
    to fetch data.
    """

    @classmethod
    async def customer_name(cls, *, customer_id: int) -> str | None:
        """
        Returns the customer name based on the ID.
        """
        if customer_id == 123:
            return 'Sid Saha'

        return None

    @classmethod
    async def customer_balance(cls, *, customer_id: int, include_pending: bool) -> float:  # pylint: disable=unused-argument
        """
        Returns the account balance of the customer based on the ID.
        """
        if customer_id == 123:
            return 123.45

        raise ValueError('Customer not found')


@dataclass
class SupportDependencies:
    """
    The dependencies needed by ths agent.
    """
    customer_id: int
    db: DatabaseConn


class SupportResult(BaseModel):
    """
    The data model produced by this agent.
    """
    support_advice: str = Field(description='The support advice returned to the customer.')
    block_card: bool = Field(description='Whether to block their card')
    risk: int = Field(
        description='Risk level of the query', ge=0, le=10,
    )


# create the agent
m = LLMModel()
m.model_type = 'ollama'
m.ollama_model_name = 'llama3.1:8b'
model = m.fetch_model()

support_agent = Agent(
    model,
    deps_type=SupportDependencies,
    result_type=SupportResult,
    system_prompt=(
        'You are a support agent in our bank, provide the the customer '
        'some support advice as per their request and judge the risk level of their query. '
        "Reply using the customer's name."
    ),
)


@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    """
    Fetches and returns the customer name.
    """
    deps: SupportDependencies = ctx.deps
    customer_name = await deps.db.customer_name(customer_id=deps.customer_id)
    return f'The name of the customer is {customer_name}'


@support_agent.tool
async def customer_balance(ctx: RunContext[SupportDependencies], include_pending: bool) -> str:
    """
    Returns the customer's current account balance.
    """
    deps: SupportDependencies = ctx.deps
    balance = await deps.db.customer_balance(
        customer_id=deps.customer_id, include_pending=include_pending,
    )
    return f'{balance:.2f}'


async def main(input_text: str, customer_id: int) -> None:
    """
    The main function to execute.
    """
    deps = SupportDependencies(
        customer_id=customer_id, db=DatabaseConn(),
    )
    result = await support_agent.run(input_text, deps=deps)

    print(result.all_messages())
    print()
    print(result.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_text', required=True, type=str,
    )
    parser.add_argument(
        '--customer_id', required=True, type=int,
    )
    args = parser.parse_args()

    asyncio.run(
        main(args.input_text, args.customer_id),
    )
