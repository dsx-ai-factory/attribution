// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"
import { Dialog as DialogPrimitive } from "radix-ui"
import { XIcon } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

// The house Dialog: the content sits inside a scrollable overlay. Tuned: the
// title is 16px at 500 instead of 18px semibold; the description's links lose
// their resting underline; open and close animations stop under reduced
// motion; the close button's name is settable (closeLabel). No shadow, as in
// house: a shadow cannot read against the black/50 scrim. Its one use here is
// Attribution's own size, so it is the only one kept (the house md size is
// dropped): a near-full-screen panel centered in the overlay, whose 92dvh
// height cannot fit between the house 100px top and bottom insets, laid out
// by its caller edge to edge; its close button sits 10px in, centered on the
// 44px header row the caller draws flush with the panel's edge. The content
// draws no outline: it is focused only as a whole (by the graph dialog as it
// opens, and by Radix when the focused control inside is removed), and a ring
// around the whole panel would point at nothing.

function Dialog({
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Root>) {
  return <DialogPrimitive.Root data-slot="dialog" {...props} />
}

function DialogTrigger({
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Trigger>) {
  return <DialogPrimitive.Trigger data-slot="dialog-trigger" {...props} />
}

function DialogContent({
  className,
  children,
  closeLabel = "Close",
  onSubmit,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Content> & {
  /** Accessible name of the close button. */
  closeLabel?: string
}) {
  return (
    <DialogPrimitive.Portal data-slot="dialog-portal">
      <DialogPrimitive.Overlay
        data-slot="dialog-overlay"
        className="fixed inset-0 isolate z-50 flex items-center justify-center overflow-y-auto overscroll-none bg-black/50 duration-100 data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0 motion-reduce:animate-none"
      >
        <DialogPrimitive.Content
          data-slot="dialog-content"
          className={cn(
            "relative flex h-[min(880px,92dvh)] w-[calc(100vw-24px)] max-w-[1440px] flex-col overflow-hidden rounded-xl bg-background text-sm ring-1 ring-foreground/10 outline-none duration-100 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95 motion-reduce:animate-none sm:w-[calc(100vw-64px)]",
            className
          )}
          {...props}
          onSubmit={(event) => {
            onSubmit?.(event)
            // The portal still bubbles React events along the React tree; keep a
            // form inside the dialog from submitting a form that wraps it.
            if (!event.isPropagationStopped()) {
              event.stopPropagation()
            }
          }}
        >
          {children}
          <DialogPrimitive.Close data-slot="dialog-close" asChild>
            <Button variant="ghost" size="icon-sm" className="absolute top-2.5 right-2.5">
              <XIcon />
              <span className="sr-only">{closeLabel}</span>
            </Button>
          </DialogPrimitive.Close>
        </DialogPrimitive.Content>
      </DialogPrimitive.Overlay>
    </DialogPrimitive.Portal>
  )
}

function DialogHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="dialog-header"
      className={cn("flex flex-col gap-2", className)}
      {...props}
    />
  )
}

function DialogTitle({
  className,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Title>) {
  return (
    <DialogPrimitive.Title
      data-slot="dialog-title"
      className={cn("text-base font-medium", className)}
      {...props}
    />
  )
}

function DialogDescription({
  className,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Description>) {
  return (
    <DialogPrimitive.Description
      data-slot="dialog-description"
      className={cn(
        "text-sm text-foreground-secondary *:[a]:hover:text-foreground",
        className
      )}
      {...props}
    />
  )
}

export {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
}
