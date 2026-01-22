#!/usr/bin/env python3
"""Test if searchagent deployment works."""

import asyncio
import subprocess


async def test():
    from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
    from openai import AsyncAzureOpenAI
    
    # Get config
    result = subprocess.run(['azd', 'env', 'get-values'], capture_output=True, text=True)
    env = {}
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            key, _, value = line.partition('=')
            env[key] = value.strip('"')
    
    endpoint = env.get('AZURE_OPENAI_ENDPOINT', '')
    deployment = 'searchagent'  # Use searchagent deployment
    
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, 'https://cognitiveservices.azure.com/.default')
    
    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version='2024-12-01-preview',
        azure_ad_token_provider=token_provider,
    )
    
    print(f'Testing {deployment}...\n')
    
    try:
        response = await client.chat.completions.create(
            model=deployment,
            messages=[
                {'role': 'system', 'content': 'You are a legal assistant specializing in UK CPR.'},
                {'role': 'user', 'content': 'What are the general rules about costs under CPR Part 44? Answer in 2 sentences with a CPR citation.'}
            ],
            max_completion_tokens=500,
        )
        print(f'✓ Success!')
        print(f'Response: {response.choices[0].message.content}')
        print(f'Tokens: {response.usage.completion_tokens}')
    except Exception as e:
        print(f'✗ Error: {e}')
    finally:
        await credential.close()


if __name__ == "__main__":
    asyncio.run(test())
