function draw_shape(s, color)
if nargin < 2
    color = 'g.';
end

pts = reshape(s, 2, [])';
plot(pts(:,1), pts(:,2), color);

for i=1:size(pts, 1)
    text(pts(i,1), pts(i,2), num2str(i));
end
end