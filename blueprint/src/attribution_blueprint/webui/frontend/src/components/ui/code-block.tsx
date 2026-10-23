// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { CheckIcon, CopyIcon } from "lucide-react"

import { cn, focusRingInset } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard"

// The house CodeBlock for recorded machine output (log lines, JSON), as plain
// text: no syntax highlighter is loaded. Tuned: the copy button is always
// shown, since a hover-only control is lost on touch and invisible to the
// keyboard until focused; its name stays fixed and a live region says
// "Copied", or "Copy failed" when the text did not reach the clipboard; its
// check is neutral. The body reads in the secondary tone
// (12.59:1), the same as log excerpts, and may wrap so a line is never cut at
// the right edge. The title is UI text, so it stays sans. The divider under
// the header belongs to the body (house draws it on the header), so a folded
// block, whose body is not rendered, has no line above its bottom border.

const CodeBlockContext = React.createContext<{ code: string } | null>(null)

function useCodeBlock() {
  const context = React.useContext(CodeBlockContext)
  if (context === null) {
    throw new Error("CodeBlock parts must be rendered inside a CodeBlock")
  }
  return context
}

function CodeBlock({
  code,
  className,
  ...props
}: React.ComponentProps<"div"> & { code: string }) {
  const value = React.useMemo(() => ({ code }), [code])
  return (
    <CodeBlockContext.Provider value={value}>
      <div
        data-slot="code-block"
        className={cn(
          "w-full overflow-hidden rounded-md border border-border bg-background text-foreground",
          className
        )}
        {...props}
      />
    </CodeBlockContext.Provider>
  )
}

function CodeBlockHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="code-block-header"
      className={cn(
        "flex items-center justify-between bg-background-active/80 px-3 py-2 text-xs text-foreground-secondary",
        className
      )}
      {...props}
    />
  )
}

function CodeBlockTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="code-block-title"
      className={cn("flex items-center gap-2", className)}
      {...props}
    />
  )
}

function CodeBlockActions({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="code-block-actions"
      className={cn("-my-1 -mr-1 flex items-center gap-2", className)}
      {...props}
    />
  )
}

function CodeBlockCopyButton({ label }: { label: string }) {
  const { code } = useCodeBlock()
  const { copied, announcement, copy } = useCopyToClipboard(2000)
  return (
    <>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={label}
        onClick={() => void copy(code)}
      >
        {copied ? <CheckIcon /> : <CopyIcon />}
      </Button>
      <span aria-live="polite" className="sr-only">
        {announcement}
      </span>
    </>
  )
}

function CodeBlockBody({
  labelledBy,
  wrap = false,
  className,
}: {
  /** Id of the title that names the scrollable region. */
  labelledBy: string
  /** Wraps long lines instead of scrolling sideways. */
  wrap?: boolean
  /** Height classes for the scroll region. */
  className?: string
}) {
  const { code } = useCodeBlock()
  return (
    <div
      role="region"
      aria-labelledby={labelledBy}
      tabIndex={0}
      className={cn("overflow-auto border-t", focusRingInset, className)}
    >
      <pre
        className={cn(
          "m-0 px-3 py-2 font-mono text-xs text-foreground-secondary",
          wrap && "break-words whitespace-pre-wrap"
        )}
      >
        {code}
      </pre>
    </div>
  )
}

export {
  CodeBlock,
  CodeBlockHeader,
  CodeBlockTitle,
  CodeBlockActions,
  CodeBlockCopyButton,
  CodeBlockBody,
}
