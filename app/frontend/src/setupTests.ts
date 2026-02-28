import '@testing-library/jest-dom';

// Polyfill ResizeObserver for jsdom (needed by FluentUI v9 components like MessageBar)
if (typeof globalThis.ResizeObserver === 'undefined') {
    globalThis.ResizeObserver = class ResizeObserver {
        observe() {}
        unobserve() {}
        disconnect() {}
    } as any;
}
