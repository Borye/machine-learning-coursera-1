% submitWeb Creates files from your code and output for web submission.
%
%   If the submit function does not work for you, use the web-submission mechanism.
%   Call this function to produce a file for the part you wish to submit. Then,
%   submit the file to the class servers using the "Web Submission" button on the 
%   Programming Exercises page on the course website.
%
%   You should call this function without arguments (submitWeb), to receive
%   an interactive prompt for submission; optionally you can call it with the partID
%   if you so wish. Make sure your working directory is set to the directory 
%   containing the submitWeb.m file and your assignment files.

function submitWeb(partId)
  if ~exist('partId', 'var') || isempty(partId)
    partId = [];
  end
  
  submit(partId, 1);
end

