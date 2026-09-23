# API authentication and limits

The API supports personal access tokens and OAuth 2.0 bearer tokens. Send tokens in the Authorization header.

The Starter plan permits 100 requests per minute. Requests above that limit receive an HTTP 429 response. Clients should wait and retry using exponential backoff.

GraphQL subscriptions are not available on the platform.
