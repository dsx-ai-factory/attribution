import { useCallback, useMemo, useRef } from 'react'
import Editor, { type OnMount, loader } from '@monaco-editor/react'
import * as monaco from 'monaco-editor'
import dayjs from 'dayjs'
import utc from 'dayjs/plugin/utc'
import type { Run } from '@/interfaces/attribution'
import { VerdictPill } from '@/components/chip'

dayjs.extend(utc)

// Make @monaco-editor/react use the bundled monaco instance instead of loading from CDN.
loader.config({ monaco })

// Register a tiny custom dark theme aligned with the Attribution palette.
let themeRegistered = false
function ensureTheme(m: typeof monaco) {
  if (themeRegistered) return
  m.editor.defineTheme('attribution-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [],
    colors: {
      'editor.background':                 '#0a0a0a',
      'editor.foreground':                 '#b6bdb8',
      'editor.lineHighlightBackground':    '#0a0a0a',
      'editor.lineHighlightBorder':        '#0a0a0a',
      'editorLineNumber.foreground':       '#5b6560',
      'editorLineNumber.activeForeground': '#7a8580',
      'editorCursor.foreground':           '#76b900',
      'editor.selectionBackground':        '#76b90033',
      'scrollbarSlider.background':        '#2a302d',
      'scrollbarSlider.hoverBackground':   '#3a423d',
      'scrollbarSlider.activeBackground':  '#5b6560',
    },
  })
  themeRegistered = true
}

interface FormattedLine {
  text: string
  isSymptom: boolean
  isError: boolean
  isDebug: boolean
}

function formatLog(run: Run): { content: string; lines: FormattedLine[]; symptomLine: number } {
  let symptomLine = -1
  const lines: FormattedLine[] = run.log.map((l, i) => {
    const ts = dayjs(l.t).utc().format('HH:mm:ss')
    const lvl =
      l.level === 'symptom' ? '◆ SYMP' :
      l.level === 'error'   ? 'ERROR ' :
      l.level === 'debug'   ? 'DEBUG ' :
                              'INFO  '
    if (l.level === 'symptom') symptomLine = i + 1
    return {
      text: `${ts}  ${lvl}  ${l.text}`,
      isSymptom: l.level === 'symptom',
      isError: l.level === 'error',
      isDebug: l.level === 'debug',
    }
  })
  return {
    content: lines.map((l) => l.text).join('\n'),
    lines,
    symptomLine,
  }
}

const MONACO_OPTIONS: monaco.editor.IStandaloneEditorConstructionOptions = {
  readOnly: true,
  domReadOnly: true,
  fontFamily: "'JetBrains Mono Variable', ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace",
  fontSize: 12,
  lineHeight: 18,
  lineNumbers: 'on',
  lineNumbersMinChars: 5,
  lineDecorationsWidth: 14,
  glyphMargin: false,
  folding: false,
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  scrollbar: { vertical: 'auto', horizontal: 'auto', verticalScrollbarSize: 8, horizontalScrollbarSize: 8 },
  overviewRulerLanes: 0,
  overviewRulerBorder: false,
  hideCursorInOverviewRuler: true,
  renderLineHighlight: 'none',
  selectionHighlight: false,
  occurrencesHighlight: 'off',
  contextmenu: false,
  fixedOverflowWidgets: true,
  wordWrap: 'off',
  guides: { indentation: false, highlightActiveIndentation: false, bracketPairs: false },
  renderWhitespace: 'none',
  padding: { top: 8, bottom: 8 },
}

function MonacoLog({ run }: { run: Run }) {
  const { content, lines, symptomLine } = useMemo(() => formatLog(run), [run])
  const decorationsRef = useRef<monaco.editor.IEditorDecorationsCollection | null>(null)

  const onMount: OnMount = useCallback(
    (editor, m) => {
      ensureTheme(m)
      m.editor.setTheme('attribution-dark')

      // Build line decorations.
      const decos: monaco.editor.IModelDeltaDecoration[] = []
      lines.forEach((l, i) => {
        const ln = i + 1
        if (l.isSymptom) {
          decos.push({
            range: new m.Range(ln, 1, ln, 1),
            options: {
              isWholeLine: true,
              className: 'attr-log-symptom',
              linesDecorationsClassName: 'attr-log-symptom-stripe',
            },
          })
        } else if (l.isError) {
          decos.push({
            range: new m.Range(ln, 1, ln, 1),
            options: { isWholeLine: true, className: 'attr-log-error' },
          })
        } else if (l.isDebug) {
          decos.push({
            range: new m.Range(ln, 1, ln, 1),
            options: { isWholeLine: true, className: 'attr-log-debug' },
          })
        }
      })
      decorationsRef.current = editor.createDecorationsCollection(decos)

      // Scroll the symptom line into view (center).
      if (symptomLine > 0) {
        editor.revealLineInCenter(symptomLine, m.editor.ScrollType.Immediate)
      }
    },
    [lines, symptomLine],
  )

  return (
    <div className="max-h-96 overflow-hidden">
      <Editor
        height="384px"
        theme="attribution-dark"
        defaultLanguage="plaintext"
        value={content}
        options={MONACO_OPTIONS}
        onMount={onMount}
      />
    </div>
  )
}

function LogAttribution({ run }: { run: Run }) {
  const matchedAt = dayjs(run.rootSymptom.time)
  const matchedAtText = matchedAt.isValid() ? matchedAt.utc().format('HH:mm:ss') : 'Unavailable'

  return (
    <div className="flex items-start gap-3 px-4 py-2.5 border-t border-border bg-background-secondary font-mono text-xs text-foreground-secondary">
      <span className="text-foreground-muted font-mono pt-px">└─</span>
      <div className="flex-1 flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <span className="inline-block bg-background-muted border border-border px-1.5 py-0.5 rounded-sm text-primary">
          {run.rootSymptom.rule}
        </span>
        <span className="text-foreground-muted">
          matched at {matchedAtText}{' '}
          {matchedAt.isValid() && <span className="text-[10px] tracking-wide">UTC</span>}
        </span>
        <span className="text-foreground-muted">·</span>
        <span className="text-foreground">"{run.rootSymptom.symptom}"</span>
      </div>
    </div>
  )
}

const OUTCOME_HEAD: Record<Run['outcome'], string> = {
  hardware: 'HARDWARE',
  software: 'SOFTWARE',
  multi:    'MULTI-CATEGORY',
  user:     'USER',
  undetermined: 'UNDETERMINED',
  pass:     'PASS',
}

export function LogHero({ run }: { run: Run }) {
  return (
    <div className="bg-background-secondary border border-border rounded-md overflow-hidden">
      <div className="flex items-center gap-3 px-4 py-3 border-b border-border bg-background-active">
        <VerdictPill outcome={run.outcome} label={run.verdictLabel} />
        <div className="text-sm text-foreground flex-1 min-w-0">
          <span className="font-mono text-foreground-muted text-xs uppercase tracking-widest mr-2">
            {OUTCOME_HEAD[run.outcome]}
          </span>
          <span>{run.detail}</span>
        </div>
      </div>
      <MonacoLog run={run} />
      <LogAttribution run={run} />
    </div>
  )
}
