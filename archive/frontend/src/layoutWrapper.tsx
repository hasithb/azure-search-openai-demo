import React, { useEffect, useState } from "react";
import { AccountInfo, EventType, PublicClientApplication } from "@azure/msal-browser";
import { checkLoggedIn, msalConfig, useLogin } from "./authConfig";
import { MsalProvider } from "@azure/msal-react";
import { LoginContext } from "./loginContext";
import Layout from "./pages/layout/Layout";

const LayoutWrapper = () => {
    const [loggedIn, setLoggedIn] = useState(false);
    const [msalInstance, setMsalInstance] = useState<PublicClientApplication | null>(null);
    const [isInitialized, setIsInitialized] = useState(false);

    useEffect(() => {
        if (!useLogin) {
            setIsInitialized(true);
            return;
        }

        let mounted = true;

        const initializeMsal = async () => {
            try {
                const instance = new PublicClientApplication(msalConfig);

                // Initialize the library
                // @ts-ignore - initialize exists on PublicClientApplication in msal-browser v4.19+
                await (instance as any).initialize();

                if (!mounted) return;

                // Default to using the first account if no account is active on page load
                if (!instance.getActiveAccount() && instance.getAllAccounts().length > 0) {
                    // Account selection logic is app dependent. Adjust as needed for different use cases.
                    instance.setActiveAccount(instance.getAllAccounts()[0]);
                }

                // Listen for sign-in event and set active account
                instance.addEventCallback(event => {
                    if (event.eventType === EventType.LOGIN_SUCCESS && event.payload) {
                        const account = event.payload as AccountInfo;
                        instance.setActiveAccount(account);
                    }
                });

                const logged = await checkLoggedIn(instance);

                if (mounted) {
                    setMsalInstance(instance);
                    setLoggedIn(logged);
                    setIsInitialized(true);
                }
            } catch (e) {
                // Initialization failure should not crash the app; log for debugging
                // eslint-disable-next-line no-console
                console.error("MSAL initialize failed", e);
                if (mounted) {
                    setIsInitialized(true);
                }
            }
        };

        initializeMsal();

        return () => {
            mounted = false;
        };
    }, []);

    // Show loading state while MSAL is initializing
    if (useLogin && !isInitialized) {
        return <div>Loading...</div>;
    }

    if (useLogin && msalInstance) {
        return (
            <MsalProvider instance={msalInstance}>
                <LoginContext.Provider
                    value={{
                        loggedIn,
                        setLoggedIn
                    }}
                >
                    <Layout />
                </LoginContext.Provider>
            </MsalProvider>
        );
    } else {
        return (
            <LoginContext.Provider
                value={{
                    loggedIn,
                    setLoggedIn
                }}
            >
                <Layout />
            </LoginContext.Provider>
        );
    }
};

export default LayoutWrapper;
