import { DefaultButton } from "@fluentui/react";
import { useMsal } from "@azure/msal-react";
import { useTranslation } from "react-i18next";

import styles from "./LoginButton.module.css";
import { getRedirectUri, loginRequest, appServicesLogout, getUsername, checkLoggedIn } from "../../authConfig";
import { useState, useEffect, useContext } from "react";
import { LoginContext } from "../../loginContext";

export const LoginButton = () => {
    const { instance } = useMsal();
    const { loggedIn, setLoggedIn } = useContext(LoginContext);
    const [username, setUsername] = useState("");
    const { t } = useTranslation();

    useEffect(() => {
        const fetchUsername = async () => {
            try {
                setUsername((await getUsername(instance)) ?? "");
            } catch (error) {
                console.error("Error fetching username:", error);
                setUsername("");
            }
        };

        fetchUsername();
    }, [instance]);

    const handleLoginPopup = () => {
        /**
         * When using popup and silent APIs, we recommend setting the redirectUri to a blank page or a page
         * that does not implement MSAL. Keep in mind that all redirect routes must be registered with the application
         * For more information, please follow this link: https://github.com/AzureAD/microsoft-authentication-library-for-js/blob/dev/lib/msal-browser/docs/login-user.md#redirecturi-considerations
         */
        instance
            .loginPopup({
                ...loginRequest,
                redirectUri: getRedirectUri()
            })
            .catch(error => console.log(error))
            .then(async () => {
                try {
                    setLoggedIn(await checkLoggedIn(instance));
                    setUsername((await getUsername(instance)) ?? "");
                } catch (error) {
                    console.error("Error updating login state:", error);
                }
            });
    };

    const handleLogoutPopup = () => {
        try {
            const activeAccount = instance.getActiveAccount();
            if (activeAccount) {
                instance
                    .logoutPopup({
                        mainWindowRedirectUri: "/", // redirects the top level app after logout
                        account: instance.getActiveAccount()
                    })
                    .catch(error => console.log(error))
                    .then(async () => {
                        try {
                            setLoggedIn(await checkLoggedIn(instance));
                            setUsername((await getUsername(instance)) ?? "");
                        } catch (error) {
                            console.error("Error updating logout state:", error);
                        }
                    });
            } else {
                appServicesLogout();
            }
        } catch (error) {
            console.error("Error during logout:", error);
            appServicesLogout();
        }
    };

    return (
        <DefaultButton
            text={loggedIn ? `${t("logout")}\n${username}` : `${t("login")}`}
            className={styles.loginButton}
            onClick={loggedIn ? handleLogoutPopup : handleLoginPopup}
        ></DefaultButton>
    );
};
