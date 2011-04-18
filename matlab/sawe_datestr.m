% Octave doesn't support FFF in datestr format strings but MATLAB does
function str=sawe_datestr(t,f)

isoctave=0~=exist('OCTAVE_VERSION','builtin');

if ~isoctave
    str = datestr(t, f);
	return;
end

strs = strsplit(f,'F');
str = [];
digits=0;
for k = 1:numel(strs)
  g = strs{k};
  if isempty(g)
    digits = digits + 1;
  else
    digits = 0;
  end

  if ~isempty(g) || k==numel(strs)
    if digits>0
      b = ((t-floor(t))*24*3600);
      c = sprintf(['%0.' num2str(digits) 'f' ], b-floor(b));
      p = c(3:end);
    else
      p = datestr(t, g);
    end
    str = [ str p ];
  end
end

