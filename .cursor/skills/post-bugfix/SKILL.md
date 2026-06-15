---
name: post-bugfix
description: Post-bugfix lifecycle checklist after fixing a bug.
---

# Post-bugfix Checklist

After fixing a bug, preserve the minimal reproduction, document the root cause, add the regression test that fails before the fix, remove temporary debugging noise, and verify the fix does not merely mask the failure. If the cause is uncertain, keep investigating.
