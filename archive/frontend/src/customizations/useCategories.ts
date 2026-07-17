import { useState, useEffect } from "react";

export interface Category {
    key: string;
    text: string;
    count?: number;
}

export function useCategories() {
    const [categories, setCategories] = useState<Category[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let mounted = true;

        const fetchCategories = async () => {
            try {
                const response = await fetch("/api/categories");
                if (!response.ok) throw new Error("Failed to fetch categories");

                const data = await response.json();
                if (mounted) {
                    setCategories(data.categories || []);
                    setError(null);
                }
            } catch (err) {
                if (mounted) {
                    setError(err instanceof Error ? err.message : "Unknown error");
                    // Set default sources on error
                    setCategories([{ key: "", text: "All Sources" }]);
                }
            } finally {
                if (mounted) setLoading(false);
            }
        };

        fetchCategories();

        return () => {
            mounted = false;
        };
    }, []);

    return { categories, loading, error };
}
