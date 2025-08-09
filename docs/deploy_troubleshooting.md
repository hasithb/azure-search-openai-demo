# RAG chat: Troubleshooting deployment

If you are experiencing an error when deploying the RAG chat solution using the [deployment steps](../README.md#deploying), this guide will help you troubleshoot common issues.

1. You're attempting to create resources in regions not enabled for Azure OpenAI (e.g. East US 2 instead of East US), or where the model you're trying to use isn't enabled. See [this matrix of model availability](https://aka.ms/oai/models).

1. You've exceeded a quota, most often number of resources per region. See [this article on quotas and limits](https://aka.ms/oai/quotas).

1. You're getting "same resource name not allowed" conflicts. That's likely because you've run the sample multiple times and deleted the resources you've been creating each time, but are forgetting to purge them. Azure keeps resources for 48 hours unless you purge from soft delete. See [this article on purging resources](https://learn.microsoft.com/azure/cognitive-services/manage-resources?tabs=azure-portal#purge-a-deleted-resource).

1. You see `CERTIFICATE_VERIFY_FAILED` when the `prepdocs.py` script runs. That's typically due to incorrect SSL certificates setup on your machine. Try the suggestions in this [StackOverflow answer](https://stackoverflow.com/a/43855394).

1. After running `azd up` and visiting the website, you see a '404 Not Found' in the browser. Wait 10 minutes and try again, as it might be still starting up. Then try running `azd deploy` and wait again. If you still encounter errors with the deployed app and are deploying to App Service, consult the [guide on debugging App Service deployments](/docs/appservice.md). Please file an issue if the logs don't help you resolve the error.

## Troubleshooting HTTP 500 errors after deployment

If you see a **500 Internal Server Error** after deploying with `azd deploy`:

1. **Wait 10 minutes** after deployment, as the app may take time to start.
2. **Check deployment logs** in the Azure Portal under your App Service's "Deployment Center" > "Logs". Look for failed deployments or errors during the build.
3. **Check application logs** using "Advanced Tools" (Kudu) in the App Service. Download the latest Docker logs and look for Python errors or missing dependencies.
4. **Check Azure Monitor/Application Insights** for exceptions and server errors.
5. **Redeploy** with `azd deploy` if you suspect a transient issue.
6. For detailed steps, see [Debugging the app on App Service](appservice.md).

Common causes include:
- Missing or incorrect Python dependencies.
- Syntax or runtime errors in your code.
- Incorrect startup command or misconfiguration.
- Environment variables not set correctly.

Review the logs for specific error messages to help diagnose the problem.
