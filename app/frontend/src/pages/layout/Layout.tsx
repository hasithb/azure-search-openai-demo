import { Outlet, Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useState } from "react";
import styles from "./Layout.module.css";

import { useLogin } from "../../authConfig";

import { LoginButton } from "../../components/LoginButton";
// CUSTOM: Help & About panel from merge-safe customizations
import { HelpAboutPanel } from "../../customizations/HelpAboutPanel";
// CUSTOM: Animated splash screen on first load
import { SplashScreen } from "../../customizations/SplashScreen";

const Layout = () => {
    const { t } = useTranslation();
    const [splashComplete, setSplashComplete] = useState(false);

    return (
        <div className={styles.layout}>
            {/* CUSTOM: Animated intro splash screen */}
            {!splashComplete && <SplashScreen onComplete={() => setSplashComplete(true)} />}

            <header className={styles.header} role={"banner"}>
                <div className={styles.headerContainer}>
                    <Link to="/" className={styles.headerTitleContainer}>
                        <h3 className={styles.headerTitle}>{t("headerTitle")}</h3>
                    </Link>
                    <div className={styles.loginMenuContainer}>
                        <HelpAboutPanel />
                        {useLogin && <LoginButton />}
                    </div>
                </div>
            </header>

            <main className={styles.main} id="main-content">
                <Outlet />
            </main>
        </div>
    );
};

export default Layout;
