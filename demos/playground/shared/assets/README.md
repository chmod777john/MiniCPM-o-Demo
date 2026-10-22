# Design assets

Exported from the user-provided Figma design through its authorized MCP connection on 2026-10-22.
File key: `wryplKDP6H9fdIbgdAArFX`; desktop voice frame: `24:3151`.

| Local asset            | Figma node          | Use                                                                |
| ---------------------- | ------------------- | ------------------------------------------------------------------ |
| `figma-brand.svg`      | `24:3177`           | Brand mark, assistant avatar, favicon                              |
| `figma-orb.png`        | `24:3583`           | Welcome and voice visualization                                    |
| `figma-microphone.svg` | `24:3302`           | Start conversation button                                          |
| `figma-layout.svg`     | `24:3197`           | Card layout navigation                                             |
| `figma-document.svg`   | `24:3201`           | Documents and text suggestion                                      |
| `figma-settings.svg`   | `24:3221`           | Developer settings                                                 |
| `figma-palette.svg`    | `24:3281`–`24:3284` | Image suggestion; original paths assembled in relative coordinates |
| `figma-code.svg`       | `24:3292`–`24:3294` | Code suggestion; original paths assembled in relative coordinates  |

Monochrome exports are used as CSS masks so selected and default states share one geometry. Additional case icons use Lucide at 1.5 px stroke, matching the reference's navigation weight. Assets are bundled locally; no Figma asset URL or OAuth credential is required at runtime.

Reference text colors: primary `#1f2937`, regular navigation `#495057`, secondary `#6c757d`, selected navigation `#1e40af`; title gradient `#1f2937` → `#58759d`. Reference fonts are Inter and MiSans VF. The app prefers those fonts when locally available, with CJK system fallbacks; those font binaries have not been bundled.
