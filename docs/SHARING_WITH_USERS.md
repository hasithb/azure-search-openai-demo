# Sharing Civil Procedure Copilot with Users

This guide explains how to share the **Civil Procedure Copilot** (AI-powered legal research assistant) with users while maintaining proper security controls.

## 📚 Application Overview

**Civil Procedure Copilot** is an AI-powered research assistant for searching:

- Civil Procedure Rules (Parts 1-89) and Practice Directions
- Commercial Court Guide (11th Edition, July 2023)
- King's Bench Division Guide (2025 Edition)
- Chancery Guide (2022 Edition)
- Patents Court Guide (February 2025)
- Technology & Construction Court Guide (October 2022)
- Circuit Commercial Court Guide (August 2023)

## Current Security Configuration

Your application is configured with the following security settings:

| Setting | Value | Description |
|---------|-------|-------------|
| `AZURE_USE_AUTHENTICATION` | `true` | Microsoft Entra ID login is enabled |
| `AZURE_ENFORCE_ACCESS_CONTROL` | `true` | Document-level access control is enforced |
| `AZURE_ENABLE_UNAUTHENTICATED_ACCESS` | `true` | Users can browse without login (but cannot search documents) |

**Application URL:** https://capps-backend-ot6tupm5qi5wy.delightfulground-1a2f1220.eastus2.azurecontainerapps.io

***

## Option 1: Add Users to Your Azure AD Tenant (Recommended for Controlled Access)

This is the most secure option for sharing with specific lawyers or team members.

### Step 1: Add the User as a Guest in Microsoft Entra Admin Center

1. Go to [Microsoft Entra Admin Center](https://entra.microsoft.com)
1. Sign in with your admin account (hbalapatabendi@gmail.com)
1. Navigate to **Users** → **All users**
1. Click **New user** → **Invite external user**
1. Enter:
   - **Email:** The user's email address
   - **Display name:** Their name
   - **Send invite message:** ✅ Check this box and customize:

     > "You've been invited to access Civil Procedure Copilot, our AI-powered legal research assistant for CPR, Practice Directions, and Court Guides. After accepting, visit https://capps-backend-ot6tupm5qi5wy.delightfulground-1a2f1220.eastus2.azurecontainerapps.io"

1. Click **Review + invite**

The user will receive an email invitation to join your tenant.

### Step 2: Grant API Permissions (Required Once)

First-time users may see "Consent Required" or "Admin approval required" errors. To fix:

1. Navigate to [Microsoft Entra Admin Center](https://entra.microsoft.com)
1. Go to **Enterprise applications**
1. Find your Client App: `Azure Search OpenAI Chat Web App` (ID: `1d382868-51d6-4200-a4ba-3a7b94ecb2d3`)
1. Go to **Permissions** in the left menu
1. Click **Grant admin consent for Default Directory**
1. Confirm when prompted

### Step 3: Configure Document Access

Since `AZURE_ENFORCE_ACCESS_CONTROL=true`, users can only see documents they have explicit access to.

**Option A: Grant Access to ALL Documents (Simplest)**

```bash
source .venv/bin/activate
python ./scripts/manageacl.py --acl-action enable_global_access
```

**Option B: Grant Specific User Access**

```bash
# Activate Python environment
source .venv/bin/activate

# Find the user's Object ID in Microsoft Entra Admin Center → Users → [User] → Overview
# Add user's Object ID to all documents in the content container
python ./scripts/manageacl.py -v --acl-type oids --acl-action add \
  --acl <USER-OBJECT-ID> \
  --url https://stgz2m4s637t5me.blob.core.windows.net/content/
```

To find a user's Object ID:

1. Go to [Microsoft Entra Admin Center](https://entra.microsoft.com) → **Users**
1. Click on the user
1. Copy the **Object ID** from the Overview page

***

## Option 2: Create a Security Group for Multiple Users

Better for managing multiple users with the same access level.

### Step 1: Create a Security Group

1. Go to [Microsoft Entra Admin Center](https://entra.microsoft.com)
1. Navigate to **Groups** → **All groups**
1. Click **New group**
   - Group type: **Security**
   - Group name: `Civil Procedure Copilot Users`
   - Membership type: **Assigned**
1. Click **Create**

### Step 2: Add Members to the Group

1. Open the group you created
1. Go to **Members** → **+ Add members**
1. Search for and select the users to add
1. Click **Select**

### Step 3: Grant Group Access to Documents

```bash
source .venv/bin/activate

# Add the group's Object ID to all documents
python ./scripts/manageacl.py -v --acl-type groups --acl-action add \
  --acl <GROUP-OBJECT-ID> \
  --url https://stgz2m4s637t5me.blob.core.windows.net/content/
```

***

## Option 3: Enable Global Document Access (Simplest)

If you want all authenticated users to access all documents:

```bash
source .venv/bin/activate
python ./scripts/manageacl.py --acl-action enable_global_access
```

This sets the special `["all"]` value on documents, making them available to any authenticated user.

***

## User Experience Flow

### What Users Will See

1. **First Visit:** User navigates to https://capps-backend-ot6tupm5qi5wy.delightfulground-1a2f1220.eastus2.azurecontainerapps.io
1. **Login Prompt:** User clicks "Login" button in the top-right corner
1. **Microsoft Sign-In:** User signs in with their Microsoft account
1. **Consent Screen:** First-time users may see a consent dialog asking for permissions:
   - "Access Azure Search OpenAI Chat API"
   - "Read your profile"
1. **Access Granted:** User can now use Civil Procedure Copilot to query CPR, Practice Directions, and Court Guides

### Key Features to Highlight

| Feature | Description |
|---------|-------------|
| **🏷️ Source Filter** | Dropdown to filter by CPR & Practice Directions or specific Court Guides (e.g., Commercial Court Guide) |
| **🔍 Search Depth** | Quick/Standard/Thorough search options depending on query complexity |
| **📖 Citations [1][2][3]** | Numbered citations link to source documents |
| **📄 Supporting Content** | Click citations to see exact source passages for verification |
| **👍👎 Feedback** | Rate responses to help improve accuracy |
| **❓ Help Button** | Blue ? button in bottom-right corner for detailed help |

### If Users Cannot Login

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| "Admin approval required" | Tenant admin needs to grant consent for the app |
| "User not found" | Invite the user as a guest to your tenant |
| "No access to documents" | Add user/group to document ACLs or enable global access |
| "AADSTS65001 consent error" | Grant admin consent in Enterprise Applications |

***

## Quick Reference: Azure Portal URLs

| Task | URL |
|------|-----|
| **Manage Users** | https://entra.microsoft.com/#view/Microsoft_AAD_UsersAndTenants/UserManagementMenuBlade/~/AllUsers |
| **Manage Groups** | https://entra.microsoft.com/#view/Microsoft_AAD_IAM/GroupsManagementMenuBlade/~/AllGroups |
| **Enterprise Apps** | https://entra.microsoft.com/#view/Microsoft_AAD_IAM/StartboardApplicationsMenuBlade/~/AppAppsPreview |
| **App Registrations** | https://entra.microsoft.com/#view/Microsoft_AAD_RegisteredApps/ApplicationMenuBlade/~/Overview/appId/1d382868-51d6-4200-a4ba-3a7b94ecb2d3 |
| **Storage Account** | https://portal.azure.com/#resource/subscriptions/0d1ec78c-510f-4a29-b851-be9a980219cb/resourceGroups/rg-cpr-rag/providers/Microsoft.Storage/storageAccounts/stgz2m4s637t5me |

***

## Security Best Practices

1. **Use Groups, Not Individual Permissions** - Easier to manage at scale
1. **Review Access Regularly** - Audit who has access to the application
1. **Use Document-Level ACLs** - Restrict sensitive documents to specific users/groups
1. **Monitor Sign-In Logs** - Check Microsoft Entra sign-in logs for unusual activity
1. **Rotate App Secrets** - Periodically rotate `AZURE_SERVER_APP_SECRET` and `AZURE_CLIENT_APP_SECRET`

***

## Revoking Access

### Remove a Specific User

1. Go to [Microsoft Entra Admin Center](https://entra.microsoft.com) → **Enterprise applications**
1. Select `Azure Search OpenAI Chat Web App`
1. Go to **Users and groups**
1. Remove the user from the list

### Remove a User from a Group

1. Go to [Microsoft Entra Admin Center](https://entra.microsoft.com) → **Groups**
1. Select the group
1. Go to **Members** → Remove the user

### Remove Document Access

```bash
source .venv/bin/activate

# Remove specific user's access to all documents
python ./scripts/manageacl.py -v --acl-type oids --acl-action remove \
  --acl <USER-OBJECT-ID> \
  --url https://stgz2m4s637t5me.blob.core.windows.net/content/

# Remove all user access to a specific document
python ./scripts/manageacl.py -v --acl-type oids --acl-action remove_all \
  --url https://stgz2m4s637t5me.blob.core.windows.net/content/<document.pdf>
```

***

## Appendix: Environment Variable Reference

| Variable | Current Value | Description |
|----------|---------------|-------------|
| `AZURE_AUTH_TENANT_ID` | `3bfe16b2-5fcc-4565-b1f1-15271d20fecf` | Azure AD Tenant (Default Directory) |
| `AZURE_SERVER_APP_ID` | `0e5b8dce-464a-42f8-aa39-1044341cf90b` | Server API app registration |
| `AZURE_CLIENT_APP_ID` | `1d382868-51d6-4200-a4ba-3a7b94ecb2d3` | Client UI app registration |
| `AZURE_USE_AUTHENTICATION` | `true` | Authentication enabled |
| `AZURE_ENFORCE_ACCESS_CONTROL` | `true` | Document ACLs enforced |
| `AZURE_ENABLE_UNAUTHENTICATED_ACCESS` | `true` | Allows browsing without login |
| `AZURE_ENABLE_HOST_UNAUTHENTICATED_ACCESS` | `true` | Lets the host serve the shell without login while backend APIs can still require auth |

***

## Related Documentation

- [Microsoft Entra ID Guest Access](https://learn.microsoft.com/entra/external-id/what-is-b2b)
- [Managing Groups in Azure AD](https://learn.microsoft.com/entra/fundamentals/how-to-manage-groups)
- [Azure AI Search Document-Level Security](https://learn.microsoft.com/azure/search/search-security-trimming-for-azure-search)
