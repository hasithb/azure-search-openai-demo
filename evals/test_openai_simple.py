#!/usr/bin/env python3
"""Simple test of OpenAI connection."""

import asyncio
import subprocess


async def test_openai():
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
    deployment = env.get('AZURE_OPENAI_CHATGPT_DEPLOYMENT', 'gpt-4o')
    api_version = env.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
    
    print(f'Endpoint: {endpoint}')
    print(f'Deployment: {deployment}')
    print(f'API Version: {api_version}')
    
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, 'https://cognitiveservices.azure.com/.default')
    
    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )
    
    # Simple test
    print('\nSending simple test message...')
    try:
        response = await client.chat.completions.create(
            model=deployment,
            messages=[{'role': 'user', 'content': 'Say hello in 3 words'}],
            max_tokens=50,
            temperature=0,
            timeout=10.0,
        )
        
        print(f'✓ Response: {response.choices[0].message.content}')
    except Exception as e:
        print(f'✗ Error: {e}')
    finally:
        await credential.close()


if __name__ == "__main__":
    asyncio.run(test_openai())
