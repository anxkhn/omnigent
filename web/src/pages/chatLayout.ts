export const CHAT_COLUMN_WIDTH =
  "max-w-3xl min-[1921px]:max-w-4xl min-[2561px]:max-w-[clamp(64rem,40vw,100rem)]";

/**
 * Composer popovers grow upward (`bottom-full`) toward ChatHeader
 * (`absolute z-30`, no paint). Stay in this band so they stack above the
 * transcript but never enter the header box.
 */
export const COMPOSER_POPOVER_Z = "z-20";

/** 16rem, or whatever still fits under the header + composer on a short window. */
export const COMPOSER_POPOVER_MAX_H = "max-h-[min(16rem,calc(100svh-14rem))]";
