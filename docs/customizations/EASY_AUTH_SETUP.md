# Easy Auth (Azure Built-in Authentication) Setup

This document describes how authentication and access control work in the deployed application using Azure Built-in Authentication ("Easy Auth").

## How it works

When the app is deployed with `AZURE_USE_AUTHENTICATION=true` and valid Entra ID app registration IDs, the Azure infrastructure automatically configures Easy Auth on the Container App (or App Service). This means:

1. **Unauthenticated users are redirected to Microsoft login** before the app even loads — no custom login UI is needed.
2. **Authenticated requests** carry the `x-ms-token-aad-access-token` header, which the backend reads automatically via `AuthenticationHelper.get_token_auth_header()`.
3. **Token refresh** is handled by the platform via the `/.auth/refresh` endpoint.
4. **Logout** is handled via `/.auth/logout`.

The frontend already supports Easy Auth tokens via `getAppServicesToken()` in `authConfig.ts`, which reads from the `/.auth/me` endpoint.

## Infrastructure (already configured)

The Bicep templates handle all provisioning:

- **Container Apps**: `infra/core/host/container-apps-auth.bicep` creates a `Microsoft.App/containerApps/authConfigs` resource with Azure AD as the identity provider, token store in blob storage, and `RedirectToLoginPage` for unauthenticated requests.
- **App Service**: `infra/core/host/appservice.bicep` configures `authsettingsV2` with the same behavior.
- **App Registrations**: `infra/core/security/appregistration.bicep` manages the client and server app registrations.

These modules are invoked from `infra/main.bicep` when `clientAppId` is non-empty.

## Required azd environment variables

These should already be set in your `.azure/<env>/.env` file:

```
AZURE_USE_AUTHENTICATION=true
AZURE_AUTH_TENANT_ID=<your-tenant-id>
AZURE_CLIENT_APP_ID=<client-app-registration-id>
AZURE_CLIENT_APP_SECRET=<client-app-secret>
AZURE_SERVER_APP_ID=<server-app-registration-id>
AZURE_SERVER_APP_SECRET=<server-app-secret>
```

## Restricting access to a security group

To limit app access to specific users or a security group, configure **assignment required** on the Enterprise Application:

### Step 1: Enable assignment requirement

1. Go to [Azure Portal](https://portal.azure.com) > **Microsoft Entra ID** > **Enterprise applications**
2. Find the **client** app registration (the one matching `AZURE_CLIENT_APP_ID`)
3. Go to **Properties**
4. Set **"Assignment required?"** to **Yes**
5. Click **Save**

With this enabled, only users or groups explicitly assigned to the app can access it. All others will see an `AADSTS50105` error ("You are not assigned a role for this application").

### Step 2: Assign users or a security group

1. In the same Enterprise Application, go to **Users and groups**
2. Click **+ Add user/group**
3. Under **Users**, select individual users, OR under **Groups**, select a security group
4. Under **Role**, select **Default Access** (or a custom app role if configured)
5. Click **Assign**

### Creating a security group (if needed)

1. Go to **Microsoft Entra ID** > **Groups** > **New group**
2. Set **Group type** to **Security**
3. Name it (e.g., "CPR Copilot Users")
4. Add members
5. Click **Create**
6. Then assign this group to the Enterprise Application as described above

## Local development

During local development (`npm run dev` / `quart run`), Easy Auth is not active — requests go directly to the backend without the `/.auth/*` endpoints. The app degrades gracefully:

- `getAppServicesToken()` returns `null` on `localhost`, so the app runs without authentication.
- The vite dev proxy forwards `/.auth/me` to the backend, but the backend won't have the Easy Auth middleware, so it returns a non-200 response which is handled gracefully.

To test authentication locally, deploy the app (`azd deploy`) and test against the deployed URL.

## Previous authentication approach (removed)

The app previously used a custom device code flow (`/api/device_login/*` endpoints) implemented in `app/backend/customizations/routes/device_login.py`. This has been replaced by Easy Auth because:

- Easy Auth is platform-managed (zero custom auth code to maintain)
- It handles token refresh, session management, and logout automatically
- It supports Conditional Access policies, MFA, and named locations out of the box
- Security group restrictions are configured via the Entra admin portal (no code changes)

The `device_login.py` file is retained in the codebase but is no longer imported or registered as a blueprint.
