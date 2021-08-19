< [All tutorials](./tutorials_index.html)

# How to clean up a messy Git commit history

Joe Cummings

3 August, 2021 (Last Updated: 19 August, 2021)
 
***

* [Motivation](#motivation)
* [Background](#background)
* [Cleaning up Git commits](#cleanup)
    1. [Combining multiple smaller commits into one](#combining)
    2. [Renaming a commit](#renaming)
    3. [Breaking one commit into multiple commits](#multiple)
    4. [Finishing up](#finish)
* [Conclusion](#conclusion)

***

### **Motivation** {#motivation}
Let's face it - your Git commit history may not always be the most helpful. There are poorly named commits, weird branch merges, commits that don't belong together, etc. Oftentimes, we ignore these problems in the name of efficiency.

<div style="width:100%;height:0;padding-bottom:56%;position:relative;"><iframe src="https://giphy.com/embed/QMHoU66sBXqqLqYvGO" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div>

But let's imagine we found a bug - a feature is not working as expected. If you have a clean, linear, Git history with well-named commits, finding the issue and fixing that commit or rewinding to a place before the broken commit is fairly trivial.

The steps I'll go over in this post are the tricks I use to clean up my Git history.

### Background {#background}

This post assumes some familiarity with Git; however, just for fun let's go over a bit of history.

Git is part of a group of tools that accomplish "version control". Why do we need "version control"?

1. Enabling collaboration. Codebases can be huge and software is a team sport.
2. Keeping track of changes. Code is complicated and developers edit and re-edit files constantly to get to a desired state. Version control systems track the changes efficiently.
3. Data backup.

Version control systems have been around since the [early 1970s](https://www.linkedin.com/learning/learning-software-version-control/the-history-of-version-control), but Git emerged in 2005 from Linus Torvalds and the Linux Foundation. Up until 2005, they used another version control system but the relationship soured and Linux was forced to create their own. Git is incredibly fast, simple, fully distributed, and can handle large projects (like the Linux kernel).

Some more resources on the subject can be found in the footnotes [^1],[^2].

[^1]: [A short history of Git](https://git-scm.com/book/en/v2/Getting-Started-A-Short-History-of-Git)
[^2]: [Version control](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control)


### Cleaning up Git commits {#cleanup}

Let's assume we are working on a feature on a separate branch (because we would NEVER work on `master/main` branch) and we've gotten to the point where we want to submit a [Pull Request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) and merge our changes into the `master/main` branch. I find it really useful to start by seeing where we are:

```bash
git log --graph
```

This pulls up all my commits (or as many as will fit in my screen) like the following:

```changelog
* commit 9b4b5040eef0504182dcdef7aea73334551cd3b9 (HEAD -> feature-1)
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:47:28 2021 -0400
|
|     Remove redundant test for feature-1
|
* commit f5286491c8ce06db3eb0667fc300836267cb9c9a
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:46:38 2021 -0400
|
|     Fix typo in feature-1 tests
|
* commit c4fb686736d462e0fc1e239b7da79610b76e197d
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:45:55 2021 -0400
|
|     Add feature-1 tests
|
* commit 594a2c077a469427f0418fee9cdcfa427e048d0f
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:51:00 2021 -0400
|
|     buttz
|
* commit ec09d3b36013b3140ac03de7f51e1019d23825a2
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 11:23:44 2021 -0400
|
|     Fix linting errors; add credentials
|
* commit db02d3b56013b3140ac03de7f91e1019123825a2
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 11:23:44 2021 -0400
|
|     Add feature-1
|
* commit 4a9c9b12531c7786802241d558f700d40538cad7 (master)
  Author: Joe Cummings <fake.email@gmail.com>
  Date:   Fri Aug 6 10:40:52 2021 -0400

      First commit
```

The `--graph` option gives a nice condensed and intuitive view of what changes exist in the history. We can see that there are 6 commits on my new branch `feature-1` and `master` is set to the very first commit. Now there are few things that we might want to clean up in this history before we commit it to the annals of `master/main` forever. 

#### 1. Combining multiple smaller commits into one {#combining}

Let's take a look at the latest three commits. We can see one is adding tests for our feature, one is a typo correction in that tests file, and one is deleting a test we've since deemed unnecessary. These don't need to be three separate commits - we gain no valuable information from that! So we want to combine these commits, which we can do using the [`rebase`](https://git-scm.com/docs/git-rebase) command. It's called `rebase` becasue it **RE**applies commits to the **BASE** of another tip/branch.

```bash
git rebase -i HEAD~3
```

Let's decode this command a little. The `-i` option means this is going to be "interactive", which means Git will show us a helpful editor to complete the rebase. It will use the default text editor on your computer, so for me it's [Vim](https://www.vim.org/). The `HEAD~3` means we want to move the `HEAD` (or current place in the code) back 3 commits, because that's the first commit we want to use to combine.

This command should popup an editor window that looks like the following:

```diff
pick c4fb686 Add feature-1 tests
pick f528649 Fix typo in feature-1 tests
pick 9b4b504 Remove redundant test for feature-1

# Rebase 594a2c0..9b4b504 onto 594a2c0 (3 commands)
#
# Commands:
# p, pick <commit> = use commit
# r, reword <commit> = use commit, but edit the commit message
# e, edit <commit> = use commit, but stop for amending
# s, squash <commit> = use commit, but meld into previous commit
# f, fixup <commit> = like "squash", but discard this commit's log message
# x, exec <command> = run command (the rest of the line) using shell
# b, break = stop here (continue rebase later with 'git rebase --continue')
# d, drop <commit> = remove commit
# l, label <label> = label current HEAD with a name
# t, reset <label> = reset HEAD to a label
# m, merge [-C <commit> | -c <commit>] <label> [# <oneline>]
# .       create a merge commit using the original merge commit's
# .       message (or the oneline, if no original merge commit was
# .       specified). Use -c <commit> to reword the commit message.
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
```

Git provides this convenient dialog that contains all the things you may want to do with this collection of commits. Decomposing this a little more, the format at the top goes:

```diff
<GIT COMMAND> <COMMIT HASH> <COMMIT MESSAGE>

pick c4fb686 Add feature-1 tests
```

The option we want is `s` or `squash`. (You could also use `f` for `fixup`.) The caveat is that all these commits have to be consectutive. So if these commits were all over our history, we'd have to use our editor and a little Vim magic to move all these commits together. Typically, I keep the oldest one where it is and move the more recent commits back. 

> Vim tips: `Shift + V` will let you select an entire line in Vim. Then, you can use `d` to cut it, move your cursor to where you want it, and `p` to paste it.

Replace the default `pick` of the two more recent commits (`9b4b504` and `f528649`) with an `s`. Your window should now look like the following:

```diff
pick c4fb686 Add feature-1 tests
s f528649 Fix typo in feature-1 tests
s 9b4b504 Remove redundant test for feature-1

# Rebase 594a2c0..9b4b504 onto 594a2c0 (3 commands)
...
```

Then save and quit with `:wq`. You should see a confirmation page like the following, which will ask you what you want your new commit message to be.

```diff
# This is a combination of 3 commits.
# This is the 1st commit message:

Add feature-1 tests

# This is the commit message #2:

Fix typo in feature-1 tests

# This is the commit message #3:

Remove redundant test for feature-1

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# Date:      Fri Aug 6 10:45:55 2021 -0400
#
# interactive rebase in progress; onto 594a2c0
# Last commands done (3 commands done):
#    squash f528649 Fix typo in feature-1 tests
#    squash 9b4b504 Remove redundant test for feature-1
# No commands remaining.
# You are currently rebasing branch 'master' on '594a2c0'.
#
# Changes to be committed:
#       new file:   main_tests.py
#
```

Let's just keep the name `Add feature-1 test` so go ahead and delete or comment out the other commit messages and then save and quit. You see a message like:

```bash
Successfully rebased and updated refs/heads/master.
```

Now when we check the git log, we see all those commits are now just one!

```changelog
* commit 9e9a25d3f86a303a1c1dd4ebf6851cf891b25b46 (HEAD -> feature-1)
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:45:55 2021 -0400
|
|     Add feature-1 tests
|
* commit 594a2c077a469427f0418fee9cdcfa427e048d0f
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:51:00 2021 -0400
|
|     buttz
|
...
```

#### 2. Renaming a commit {#renaming}

It would appear that I was so tired/frustrated with my work that I named my commit `buttz`. Upon closer inspection of the commit, it's actually solid code, but the name is utterly useless. Let's whip out our trusty interactive rebase and fix this trangression.

```bash
git rebase -i HEAD~2
```

We only want to move the `HEAD` back 2 commits now, as that's the start of the commit we want to fix. This time replace the `pick` next to the commit named "buttz" with an `r` for `reword`.

```diff
r 594a2c0 buttz
pick 9e9a25d Add feature-1 tests

# Rebase 4a9c9b1..9e9a25d onto 4a9c9b1 (2 commands)
...
```

Upon saving and closing, this will bring up an editor that looks very similar to the one we just saw for rebasing. Instead of commenting out the old commit message or deleting it, just rename it to something more descriptive like `"Add \__pycache__ to the .gitignore"`. Upon saving and quitting, we should again see a succesful message.

```bash
Successfully rebased and updated refs/heads/master.
```

And the log now looks like the following. We're getting there!

```changelog
* commit 1e46146a23de69f48d4a9c4e734396f1fe2df0cc (HEAD -> feature-1)
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:45:55 2021 -0400
|
|     Add feature-1 tests
|
* commit 500e768614139ec1acff8316e2f28af5035e3829
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:51:00 2021 -0400
|
|     Add __pycache__ to the .gitignore
|
...
```

#### 3. Breaking one commit into multiple commits {#multiple}

This is one of the more complicated procedures. In commit `ec09d3b36013b3140ac03de7f51e1019d23825a2` we're actually doing two different things, which is *not* best practice - we need to break it up into two separate commits. 

```bash
git rebase -i HEAD~3`
```

When the window pops up, we want to use the `e` option for `edit`. This will stop you at the given commit to make edits.

```bash
Stopped at ec09d3b...  Fix linting errors; add credentials
```

There's a lot you can do here. Essentially you're back at the commit and can make whatever edits you want to! However, we want to split this commit into two commits. To do that, we need to unstage the changes from this commit. To do that we use the command:

```bash
git reset HEAD~
```

This resets the `HEAD` to what it was when the commit was started. You should see some output of which files were unstaged by this command. Now, we can use the `git add ...` command to add back the changes - this time differentiating all linting changes from the adding of the credentials.

Once you've created two new commits, you can continue with `git rebase --continue`. You should now see the successful rebase message and your log now looks like ...

```changelog
* commit 1e46146a23de69f48d4a9c4e734396f1fe2df0cc (HEAD -> feature-1)
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:45:55 2021 -0400
|
|     Add feature-1 tests
|
* commit 500e768614139ec1acff8316e2f28af5035e3829
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:51:00 2021 -0400
|
|     Add __pycache__ to the .gitignore
|
* commit c1e50d2bb286cfb316eb23ac4206ce5ec5cded58
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 11:39:03 2021 -0400
|
|     Fix linting errors
|
* commit cba7f4751f94be62b80498b594804687d364a20b
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 11:38:53 2021 -0400
|
|     Add credentials
|
* commit db02d3b56013b3140ac03de7f91e1019123825a2
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 11:23:44 2021 -0400
|
|     Add feature-1
|
* commit 4a9c9b12531c7786802241d558f700d40538cad7 (master)
  Author: Joe Cummings <fake.email@gmail.com>
  Date:   Fri Aug 6 10:40:52 2021 -0400

      First commit
```

#### 4. Finishing up {#finish}

Lastly, we want to make sure we've rebased against the `master/main` branch in case there are any changes we haven't picked up. The commands to complete this task are:

```bash
git fetch origin
git rebase origin/master
```

Once this is done, you can push your commit! Now because we've done all these changes, you'll have to commit a cardinal Git sin: force pushing. Don't worry, it's completely fine, just use the command `git push origin BRANCH --force-with-lease`. The `--force-with-lease` option is a small failsafe that will reject overwriting changes if they conflict with another authors work on the same branch, regardless of the force. You're now ready to merge into `master/main`!

If you check `git log --graph` you should now see the following:

```changelog
* commit 1e46146a23de69f48d4a9c4e734396f1fe2df0cc (HEAD -> master)
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:45:55 2021 -0400
|
|     Add feature-1 tests
|
* commit 500e768614139ec1acff8316e2f28af5035e3829
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 10:51:00 2021 -0400
|
|     Add __pycache__ to the .gitignore
|
* commit c1e50d2bb286cfb316eb23ac4206ce5ec5cded58
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 11:39:03 2021 -0400
|
|     Fix linting errors
|
* commit cba7f4751f94be62b80498b594804687d364a20b
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 11:38:53 2021 -0400
|
|     Add credentials
|
* commit db02d3b56013b3140ac03de7f91e1019123825a2
| Author: Joe Cummings <fake.email@gmail.com>
| Date:   Fri Aug 6 11:23:44 2021 -0400
|
|     Add feature-1
|
* commit 4a9c9b12531c7786802241d558f700d40538cad7
  Author: Joe Cummings <fake.email@gmail.com>
  Date:   Fri Aug 6 10:40:52 2021 -0400

      First commit
```

What a gorgeous and useful Git commit history!

### Conclusion {#conclusion}

In this article, you learned a few tips and tricks to clean up your Git commit history to make your life easier. 