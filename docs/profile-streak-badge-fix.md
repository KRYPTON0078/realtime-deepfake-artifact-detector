# Profile README streak badge fix

Your profile streak badge (`KRYPTON0078/KRYPTON0078`) can show **0** when GitHub caches a stale image from the stats API. Replace the `# 📊 GitHub Stats:` block in that README with:

```markdown
# 📊 GitHub Stats:
![](https://github-readme-stats.anuraghazra1.vercel.app/api?username=KRYPTON0078&theme=dark&hide_border=false&include_all_commits=true&count_private=true&cache_seconds=60&v=20260819054211)<br/>
![](https://github-readme-streak-stats-eight.vercel.app/?user=KRYPTON0078&theme=dark&hide_border=false&date_format=M%20j%5B%2C%20Y%5D&v=20260819054211)<br/>
![](https://github-readme-stats.anuraghazra1.vercel.app/api/top-langs/?username=KRYPTON0078&theme=dark&hide_border=false&include_all_commits=true&count_private=true&layout=compact&cache_seconds=60&v=20260819054211)
```

Or run:

```powershell
powershell -File scripts/update-profile-streak-badge.ps1
```

After pushing, hard-refresh your profile page (Ctrl+F5). The live API already reports a **401-day** streak.
