{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module Data.Accelerate.TensorFlow.Desugar where

import Data.Array.Accelerate.AST.Operation
    ( expType,
      paramsIn',
      fromGrounds,
      PrimFun(..),
      PreArgs((:>:), ArgsNil),
      ExpVars,
      PreOpenAcc(..),
      GroundR(GroundRscalar),
      Modifier(In, Mut, Out),
      PreOpenExp(..),
      PreOpenFun(Body, Lam),
      ArrayInstr(..),
      GroundVars,
      OperationAcc,
      Out,
      In,
      Arg(ArgFun, ArgArray, ArgVar), Var (..), PrimConst (..) )
import Data.Array.Accelerate.AST.LeftHandSide
    ( LeftHandSide(LeftHandSideWildcard, LeftHandSideSingle) )
import Data.Array.Accelerate.Backend
    ( DesugarAcc(mkGenerate, mkPermute, mkMap) )
import Data.Array.Accelerate.Type
    ( Word8,
      scalarTypeWord8,
      scalarTypeInt,
      IntegralType(..),
      NumType(IntegralNumType, FloatingNumType),
      SingleType(NumSingleType),
      ScalarType(SingleScalarType), BoundedType (..) )
import Data.Array.Accelerate.Representation.Array
    ( Buffer, Buffers, ArrayR(ArrayR) )
import Data.Array.Accelerate.Representation.Type
    ( Distributes(reprIsSingle),
      Distribute,
      TupR(TupRpair, TupRunit, TupRsingle),
      TypeR )
import Data.Array.Accelerate.Trafo.Var
    ( declareVars, DeclareVars(DeclareVars) )
import Data.Array.Accelerate.AST.Idx ( Idx(ZeroIdx, SuccIdx) )
import Data.Array.Accelerate.Trafo.Operation.Substitution
    ( weakenArrayInstr, aletUnique, Sink(..) )
import Data.Array.Accelerate.AST.Environment
    ( mapEnv, prj', (.>), weakenId, push', type (:>), Env(Empty) )
import Data.Array.Accelerate.Representation.Ground ( buffersR )
import Data.Array.Accelerate.Trafo.Desugar ( desugarAlloc )
import Data.Array.Accelerate.Trafo.Exp.Substitution
    ( weakenVars, SinkExp(weakenE), rebuildNoArrayInstr )
import Data.Type.Equality ( type (:~:)(Refl) )
import Data.Array.Accelerate.Representation.Shape
    ( dim1, shapeType, ShapeR(..), DIM1 )
import Prelude hiding (exp)

import Data.Accelerate.TensorFlow.Type
    ( TensorTypeDict(TensorTypeDict),
      tfTensorTypeDict,
      tfAllDict,
      tfFloatDict,
      tfIntDict,
      tfModDict,
      tfNum'Dict,
      tfNumDict,
      tfOrdDict,
      OneOfDict(OneOfDict), zero )
import Data.Accelerate.TensorFlow.Operation
    ( TensorOp(TWhere, TTensorScatter, TCast, TBooleanMask, TConstant,
               TId, TGather, TVar, TSelect, TAdd, TMul, TSub, TNeg, TAbs, TSign,
               TTruncateDiv, TTruncateMod, TBitwiseAnd, TBitwiseOr, TBitwiseXor,
               TInvert, TRealDiv, TReciprocal, TSin, TCos, TTan, TAsin, TAcos,
               TAtan, TSinh, TCosh, TTanh, TAsinh, TAcosh, TAtanh, TExp, TSqrt,
               TLog, TPow, TAtan2, TLess, TGreater, TLessEqual, TGreaterEqual,
               TEqual, TNotEqual, TMaximum, TMinimum, TLogicalAnd, TLogicalOr,
               TLogicalNot),
      ScatterFun(ScatterFunMin, ScatterFunAdd) )
import Data.Array.Accelerate.Interpreter (evalMinBound, evalPi, evalMaxBound)
import Data.Array.Accelerate.Representation.Slice

instance DesugarAcc TensorOp where
  mkMap (ArgFun (Lam lhs (Body body))) (ArgArray _ (ArrayR _ t) _ gvb) aOut =
    mkMapF (push' Empty (lhs, distributeBIdx t gvb)) body aOut
  mkMap _ _ _ = error "impossible"

  mkGenerate f (ArgArray _ (ArrayR sh t) gv gvb)
    | DeclareVars lhs w k          <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
    , DeclareVars lhs' w' k'       <- declareVars $ buffersR (TupRsingle scalarTypeInt)
    , DeclareVars lhs'' w'' k''    <- declareVars $ buffersR (TupRsingle scalarTypeInt)
    , DeclareVars lhs''' w''' k''' <- declareVars $ buffersR $ shapeType sh
    = aletUnique lhs (Compute (ShapeSize sh (paramsIn' $ fromGrounds gv))) $
      -- 1) Create a Tensor of flattened shape sh with only ones
      aletUnique lhs' (desugarAlloc (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars w $ fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (Exec
        (TConstant scalarTypeInt 1)
        (ArgArray Out (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars (w' .> w) gv) (k' weakenId) :>: ArgsNil)
      ) $
      -- 2) Obtain a tensor of indices (tf.where returns list of indices pointing to values > 0)
      aletUnique lhs'' (desugarAlloc (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars (w' .> w) $ fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (Exec
        TWhere
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeInt)) (TupRpair TupRunit (k (w'' .> w'))) (k' w'') :>:
         ArgArray Out (ArrayR dim1 (TupRsingle scalarTypeInt)) (TupRpair TupRunit (k (w'' .> w'))) (k'' weakenId) :>:
         ArgsNil)
      ) $
      -- 3) Convert 1d indices to multidimensional indices
      aletUnique lhs''' (desugarAlloc (ArrayR dim1 (shapeType sh)) (fromGrounds $ TupRpair TupRunit (k (w'' .> w')))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkMap -- vraag woensdag: dit pakt alleen het eerste element, waarom?
        (ArgFun $ Lam (LeftHandSideSingle scalarTypeInt) $ Body (FromIndex sh (paramsIn' $ fromGrounds $ weakenVars (w''' .> w'' .> w' .> w) gv) (Evar (Var scalarTypeInt ZeroIdx))))
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeInt)) (TupRpair TupRunit (k (w''' .> w'' .> w'))) (k'' w'''))
        (ArgArray Out (ArrayR dim1 (shapeType sh)) (TupRpair TupRunit (k (w''' .> w'' .> w'))) (k''' weakenId))
      )
      -- 4) Apply f to the indices
      (mkMap
        (weaken (w''' .> w'' .> w' .> w) f)
        (ArgArray In (ArrayR dim1 (shapeType sh)) (TupRpair TupRunit (k (w''' .> w'' .> w'))) (k''' weakenId))
        (ArgArray Out (ArrayR dim1 t) (TupRpair TupRunit (k (w''' .> w'' .> w'))) (weakenVars (w''' .> w'' .> w' .> w) gvb))
      )

  mkPermute
    (ArgFun comb)
    (ArgArray _ (ArrayR sh' _) gv' gvb')
    perm
    (ArgArray _ (ArrayR sh t) gv gvb)
    | maybeSh'                     <- TupRpair (TupRsingle scalarTypeWord8) (TupRpair TupRunit (shapeType sh'))
    , DeclareVars lhs w k          <- declareVars $ buffersR maybeSh'
    , DeclareVars lhs' w' k'       <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
    , DeclareVars lhs'' w'' k''    <- declareVars $ buffersR (shapeType sh')
    , DeclareVars lhs''' w''' k''' <- declareVars $ buffersR t
    = -- 1) Create an array of maybeSh' with perm
      aletUnique lhs (desugarAlloc (ArrayR sh maybeSh') (fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkGenerate
        (weaken w perm)
        (ArgArray Out (ArrayR sh maybeSh') (weakenVars w gv) (k weakenId))
      ) $
      -- 2) To apply boolean mask with 1D arrays, we need to flatten. Calculate the dim1 size.
      aletUnique lhs' (Compute (ShapeSize sh (paramsIn' $ weakenVars w $ fromGrounds gv))) $
      -- 3) Get 1D array of (Just sh'), by applying a boolean mask with predicate isJust.
      aletUnique lhs'' (desugarAlloc (ArrayR sh (shapeType sh')) (fromGrounds (weakenVars (w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask (shapeType sh')
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) (TupRpair TupRunit (k' w'')) (isJust (TupRpair TupRunit (shapeType sh')) (k (w'' .> w'))))
        (fromJust (shapeType sh') (k (w'' .> w')))
        (k'' weakenId)
      ) $
      -- 3) Get 1D array of source indices (sh) with perm output (Just sh'), by applying a boolean mask with predicate isJust.
      aletUnique lhs''' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w'' .> w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask t
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) (TupRpair TupRunit (k' (w''' .> w''))) (isJust (TupRpair TupRunit (shapeType sh')) (k (w''' .> w'' .> w'))))
        (weakenVars (w''' .> w'' .> w' .> w) gvb)
        (k''' weakenId)
      ) $
      -- 4) Apply tf.tensor_scatter
      Exec (TTensorScatter scatterFun) (
        ArgArray Mut (ArrayR sh' t) (weakenVars (w''' .> w'' .> w' .> w) gv') (weakenVars (w''' .> w'' .> w' .> w) gvb') :>:
        ArgArray In (ArrayR sh (shapeType sh')) (weakenVars (w''' .> w'' .> w' .> w) gv) (k'' w''') :>:
        ArgArray In (ArrayR sh t) (weakenVars (w''' .> w'' .> w' .> w) gv) (k''' weakenId) :>:
        ArgsNil
      )
        where scatterFun = case comb of
                Lam (LeftHandSideSingle _) (Lam (LeftHandSideSingle _) (Body (PrimApp fun (Pair (Evar (Var _ (SuccIdx ZeroIdx))) (Evar (Var _ ZeroIdx)))))) -> case fun of
                  PrimAdd _ -> ScatterFunAdd
                  PrimSub _ -> ScatterFunMin
                  _         -> error "primfun not yet supported"
                _ -> error "complex combination for permute not supported"

booleanMask :: TypeR a -> Arg env (In DIM1 Word8) -> GroundVars env (Buffers a) -> GroundVars env (Buffers a) -> PreOpenAcc TensorOp env ()
booleanMask TupRunit _ _ gvb = Return gvb
booleanMask t@(TupRsingle s) (ArgArray _ _ gv gvbIn2) gvbIn1 gvbOut
  | OneOfDict <- tfAllDict s
  = Exec (TBooleanMask s) (ArgArray In (ArrayR dim1 (TupRpair t (TupRsingle scalarTypeWord8))) gv (TupRpair gvbIn1 gvbIn2) :>: ArgArray Out (ArrayR dim1 t) gv gvbOut :>: ArgsNil)
booleanMask (TupRpair t1 t2) aOut (TupRpair gvbIn1 gvbIn2) (TupRpair gvbOut1 gvbOut2) = Alet (LeftHandSideWildcard TupRunit) TupRunit
 (booleanMask t1 aOut gvbIn1 gvbOut1)
 (booleanMask t2 aOut gvbIn2 gvbOut2)
booleanMask _ _ _ _ = error "impossible"

isJust :: TypeR a -> GroundVars env (Buffers (Word8, a)) -> GroundVars env (Buffers Word8)
isJust _ (TupRpair word8 _) = word8
isJust _ _ = error "impossible"

fromJust :: TypeR a -> GroundVars env (Buffers (Word8, ((), a))) -> GroundVars env (Buffers a)
fromJust _ (TupRpair _ (TupRpair _ a)) = a
fromJust _ _ = error "impossible"

mkMapF :: forall env env' sh t. BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' t
  -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapF _ (Const s e) aOut | OneOfDict <- tfAllDict s = Exec (TConstant s e) $ aOut :>: ArgsNil

mkMapF env (PrimApp f exp) aOut = mkMapPrimAppF f env exp aOut

mkMapF env (Let elhs exp1 exp2) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp1
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF
     (weakenEnv w env)
     (weakenArrayInstr w exp1)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   mkMapF
     (push' (weakenEnv w env) (elhs, distributeBIdx a (k weakenId)))
     (weakenArrayInstr w exp2)
     (weaken w aOut)

mkMapF env (Evar (Var st idx)) (ArgArray _ arrayR gv gvb@(TupRsingle (Var groundR _)))
  | Refl <- reprIsSingle @ScalarType @t @Buffer st
  , OneOfDict <- tfAllDict st
  = let (BIdx idx') = prj' idx env
        gvb'        = TupRsingle (Var groundR idx')
    in  Exec
          (TId st)
          (
            ArgArray In arrayR gv gvb' :>:
            ArgArray Out arrayR gv gvb :>:
            ArgsNil
          )

mkMapF env (Pair exp1 exp2) (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2))
 = Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF env exp1 (ArgArray Out (ArrayR sh t1) gv gvb1))
   (mkMapF env exp2 (ArgArray Out (ArrayR sh t2) gv gvb2))

mkMapF env (ToIndex sh exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ shapeType sh
  , DeclareVars lhs' w' k' <- declareVars $ shapeType sh
  = mkMapF env (Let lhs exp1 (Let lhs' (weakenE w exp2) (toIndex sh (k w') (k' weakenId)))) aOut

mkMapF env (ShapeSize sh exp) aOut
  | DeclareVars lhs _ k <- declareVars $ shapeType sh
  = mkMapF env (Let lhs exp (shapeSize sh (k weakenId))) aOut

mkMapF env (FromIndex sh exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ shapeType sh
  , DeclareVars lhs' w' k' <- declareVars $ TupRsingle scalarTypeInt
  = mkMapF env (Let lhs exp1 (Let lhs' (weakenE w exp2) (fromIndex sh (k w') (k' weakenId)))) aOut

mkMapF env (IndexSlice slix exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ sliceEltR slix
  , DeclareVars lhs' w' k' <- declareVars $ shapeType (sliceDomainR slix)
  = mkMapF env (Let lhs exp1 (Let lhs' (weakenE w exp2) (indexSlice slix (k w') (k' weakenId)))) aOut

mkMapF env (IndexFull slix exp1 exp2) aOut -- (kopieer vector naar matrix)
  | DeclareVars lhs w k <- declareVars $ sliceEltR slix
  , DeclareVars lhs' w' k' <- declareVars $ shapeType (sliceShapeR slix)
  = mkMapF env (Let lhs exp1 (Let lhs' (weakenE w exp2) (indexFull slix (k w') (k' weakenId)))) aOut

mkMapF env (Cond cond exp1 exp2) (ArgArray _ (ArrayR sh t) gv gvb)
  -- -| isNotLoop exp1 -- todo add preprocess check to throw an error if theres a loop, mem access, etc.
  -- , isNotLoop exp2
  | DeclareVars lhs w k <- declareVars $ buffersR $ TupRsingle scalarTypeWord8
  , DeclareVars lhs' w' k' <- declareVars $ buffersR t
  , DeclareVars lhs'' w'' k'' <- declareVars $ buffersR t
  = aletUnique lhs (desugarAlloc (ArrayR sh (TupRsingle scalarTypeWord8)) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkMapF (weakenEnv w env) (weakenArrayInstr w cond) (ArgArray Out (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars w gv) (k weakenId))) $
    aletUnique lhs' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars w gv))) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkMapF (weakenEnv (w' .> w) env) (weakenArrayInstr (w' .> w) exp1) (ArgArray Out (ArrayR sh t) (weakenVars (w' .> w) gv) (k' weakenId))) $
    aletUnique lhs'' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w' .> w) gv))) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkMapF (weakenEnv (w'' .> w' .> w) env) (weakenArrayInstr (w'' .> w' .> w) exp2) (ArgArray Out (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId)))
    (select
      t
      (ArgArray In (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars (w'' .> w' .> w) gv) (k (w'' .> w')))
      (ArgArray In (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k' w''))
      (ArgArray In (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId))
      (ArgArray Out (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (weakenVars (w'' .> w' .> w) gvb))
    )

mkMapF _ Nil _ = Return TupRunit

mkMapF env (ArrayInstr (Index var) exp) (ArgArray _ (ArrayR sh t@(TupRsingle st)) gv gvb)
  | Refl <- reprIsSingle @ScalarType @t @Buffer st
  , OneOfDict <- tfAllDict st
  , i <- expType exp
  , DeclareVars lhs w k <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
  , DeclareVars lhs' w' k' <- declareVars $ buffersR i
  =  -- To apply gather to 1d array, calculate the dim1 size
    aletUnique lhs (Compute (ShapeSize sh (paramsIn' $ fromGrounds gv))) $
    aletUnique lhs' (desugarAlloc (ArrayR sh i) (weakenVars w $ fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkMapF (weakenEnv (w' .> w) env) (weakenArrayInstr (w' .> w) exp) (ArgArray Out (ArrayR sh i) (weakenVars (w' .> w) gv) (k' weakenId)))
    (Exec
      (TGather st)
      (
        ArgArray In (ArrayR dim1 t) (TupRpair TupRunit (k w')) (TupRsingle (weaken (w' .> w) var)) :>:
        ArgArray In (ArrayR sh i) (weakenVars (w' .> w) gv) (k' weakenId) :>:
        ArgArray Out (ArrayR sh t) (weakenVars (w' .> w) gv) (weakenVars (w' .> w) gvb) :>:
        ArgsNil
      )
    )

mkMapF _ (ArrayInstr (Parameter var@(Var st _)) Nil) aOut
  | OneOfDict <- tfAllDict st
  = Exec
      (TVar st)
      (
        ArgVar (TupRsingle var) :>:
        aOut :>:
        ArgsNil
      )

mkMapF env (Undef st) aOut = mkMapF env (Const st (zero st)) aOut
mkMapF env (Coerce stIn stOut exp) (ArgArray _ (ArrayR sh t) gv gvb)
  | OneOfDict <- tfAllDict stIn
  , OneOfDict <- tfAllDict stOut
  , a <- expType exp
  , DeclareVars lhs w k <- declareVars $ buffersR a
  = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkMapF (weakenEnv w env) (weakenArrayInstr w exp) (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId)))
    (Exec
      (TCast stIn stOut)
      ( ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>:
        ArgArray Out (ArrayR sh t) (weakenVars w gv) (weakenVars w gvb) :>:
        ArgsNil)
     )

mkMapF env (PrimConst (PrimMinBound bt@(IntegralBoundedType it))) aOut = mkMapF env (Const (SingleScalarType (NumSingleType (IntegralNumType it))) (evalMinBound bt)) aOut
mkMapF env (PrimConst (PrimMaxBound bt@(IntegralBoundedType it))) aOut = mkMapF env (Const (SingleScalarType (NumSingleType (IntegralNumType it))) (evalMaxBound bt)) aOut
mkMapF env (PrimConst (PrimPi ft)) aOut                                = mkMapF env (Const (SingleScalarType (NumSingleType (FloatingNumType ft))) (evalPi ft)) aOut

mkMapF env (Foreign _ _ fallback exp) (ArgArray _ (ArrayR sh t) gv gvb)
  | a <- expType exp
  , DeclareVars lhs w k <- declareVars $ buffersR a
  = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkMapF (weakenEnv w env) (weakenArrayInstr w exp) (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId)))
    (mkMap 
      (ArgFun (rebuildNoArrayInstr fallback)) 
      (ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId))
      (ArgArray Out (ArrayR sh t) (weakenVars w gv) (weakenVars w gvb)))

-- Not supported
mkMapF _ VecPack {} _   = error "VecPack operation not supported by TensorFlow backend."
mkMapF _ VecUnpack {} _ = error "VecUnpack operation not supported by TensorFlow backend."
mkMapF _ Case {} _      = error "Case operation not supported by TensorFlow backend."
mkMapF _ While {} _     = error "While operation not supported by TensorFlow backend."

mkMapF _ _ _ = error "impossible"

-- mkMapForeign :: TypeR t -> PreOpenFun NoArrayInstr () (x -> t) -> PreOpenExp (ArrayInstr env) env' x -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
-- mkMapForeign t fun exp aOut = case fun of
--                                 Body poe -> let poe' = rebuildNoArrayInstr poe in mkMapF Empty poe' _
--                                 Lam lhs pof -> _


select :: TypeR t -> Arg env (In sh Word8) -> Arg env (In sh t) -> Arg env (In sh t) -> Arg env (Out sh t) -> PreOpenAcc TensorOp env ()
select (TupRpair t1 t2) (ArgArray _ (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray _ _ gvIn2 (TupRpair gvbIn21 gvbIn22)) (ArgArray _ _ gvIn3 (TupRpair gvbIn31 gvbIn32)) (ArgArray _ _ gvOut (TupRpair gvbOut1 gvbOut2)) =
  Alet (LeftHandSideWildcard TupRunit) TupRunit
  (select t1 (ArgArray In (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray In (ArrayR sh t1) gvIn2 gvbIn21) (ArgArray In (ArrayR sh t1) gvIn3 gvbIn31) (ArgArray Out (ArrayR sh t1) gvOut gvbOut1))
  (select t2 (ArgArray In (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray In (ArrayR sh t2) gvIn2 gvbIn22) (ArgArray In (ArrayR sh t2) gvIn3 gvbIn32) (ArgArray Out (ArrayR sh t2) gvOut gvbOut2))
select t@(TupRsingle s) (ArgArray _ (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray _ _ _ gvbIn2) (ArgArray _ _ _ gvbIn3) (ArgArray _ _ gvOut gvbOut)
  | OneOfDict <- tfAllDict s
  = Exec
    (TSelect s)
    (ArgArray In (ArrayR sh (TupRpair tWord8 (TupRpair (TupRsingle s) (TupRsingle s)))) gvIn1 (TupRpair gvbIn1 (TupRpair gvbIn2 gvbIn3)) :>:
     ArgArray Out (ArrayR sh t) gvOut gvbOut :>:
     ArgsNil)
select TupRunit _ _ _ _ = Return TupRunit
select _ _ _ _ _ = error "impossible"

mkMapPrimAppF :: PrimFun (a -> t) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' a -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapPrimAppF (PrimAdd nt)  | OneOfDict <- tfNumDict nt = mkMapPrimAppF'  $ TAdd (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimMul nt)  | OneOfDict <- tfNumDict nt = mkMapPrimAppF'  $ TMul (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimSub nt)  | OneOfDict <- tfNumDict nt = mkMapPrimAppF'  $ TSub (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimNeg nt)  | OneOfDict <- tfNum'Dict nt = mkMapPrimAppF' $ TNeg (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimAbs nt)  | OneOfDict <- tfNum'Dict nt = mkMapPrimAppF' $ TAbs (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimSig nt)  | OneOfDict <- tfNum'Dict nt = mkMapPrimAppF' $ TSign (SingleScalarType (NumSingleType nt))

mkMapPrimAppF (PrimQuot it) | OneOfDict <- tfNumDict (IntegralNumType it) = mkMapPrimAppF' $ TTruncateDiv (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimRem it)  | OneOfDict <- tfModDict it = mkMapPrimAppF' $ TTruncateMod (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimQuotRem it) = mkMapPrimAppF2' (PrimQuot it) (PrimRem it)

mkMapPrimAppF (PrimIDiv it) | OneOfDict <- tfNumDict (IntegralNumType it) = mkMapPrimAppF' $ TRealDiv (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimMod it)  | OneOfDict <- tfModDict it = mkMapPrimAppF' $ TTruncateMod (SingleScalarType (NumSingleType (IntegralNumType it)))

mkMapPrimAppF (PrimDivMod it) = mkMapPrimAppF2' (PrimIDiv it) (PrimMod it)

mkMapPrimAppF (PrimBAnd it) | OneOfDict <- tfIntDict it = mkMapPrimAppF' $ TBitwiseAnd (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimBOr it)  | OneOfDict <- tfIntDict it = mkMapPrimAppF' $ TBitwiseOr (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimBXor it) | OneOfDict <- tfIntDict it = mkMapPrimAppF' $ TBitwiseXor (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimBNot it) | OneOfDict <- tfIntDict it = mkMapPrimAppF' $ TInvert (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimBShiftL it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimBShiftR it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimBRotateL it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimBRotateR it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimPopCount it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimCountLeadingZeros it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimCountTrailingZeros it) | OneOfDict <- tfIntDict it = undefined

mkMapPrimAppF (PrimFDiv ft) | OneOfDict <- tfNumDict (FloatingNumType ft) = mkMapPrimAppF' $ TRealDiv (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimRecip ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TReciprocal (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimSin ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TSin (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimCos ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TCos (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimTan ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TTan (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAsin ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAsin (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAcos ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAcos (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAtan ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAtan (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimSinh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TSinh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimCosh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TCosh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimTanh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TTanh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAsinh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAsinh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAcosh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAcosh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAtanh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAtanh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimExpFloating ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TExp (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimSqrt ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TSqrt (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimLog ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TLog (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimFPow ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TPow (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimLogBase ft) | OneOfDict <- tfFloatDict ft = undefined -- mkMapPrimAppF' $ TLog1p (SingleScalarType (NumSingleType (FloatingNumType ft)))

mkMapPrimAppF (PrimTruncate ta tb) = undefined
mkMapPrimAppF (PrimRound ta tb) = undefined
mkMapPrimAppF (PrimFloor ta tb) = undefined
mkMapPrimAppF (PrimCeiling ta tb) = undefined

mkMapPrimAppF (PrimAtan2 ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAtan2 (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimIsNaN _) = undefined
mkMapPrimAppF (PrimIsInfinite _) = undefined

mkMapPrimAppF (PrimLt st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TLess (SingleScalarType st)
mkMapPrimAppF (PrimGt st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TGreater (SingleScalarType st)
mkMapPrimAppF (PrimLtEq st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TLessEqual (SingleScalarType st)
mkMapPrimAppF (PrimGtEq st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TGreaterEqual (SingleScalarType st)
mkMapPrimAppF (PrimEq st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TEqual (SingleScalarType st)
mkMapPrimAppF (PrimNEq st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TNotEqual (SingleScalarType st)
mkMapPrimAppF (PrimMax st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TMaximum (SingleScalarType st)
mkMapPrimAppF (PrimMin st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TMinimum (SingleScalarType st)

mkMapPrimAppF PrimLAnd = mkMapPrimAppF' TLogicalAnd
mkMapPrimAppF PrimLOr = mkMapPrimAppF' TLogicalOr
mkMapPrimAppF PrimLNot = mkMapPrimAppF' TLogicalNot

mkMapPrimAppF (PrimFromIntegral it nt)
  | TensorTypeDict <- tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType it)))
  , TensorTypeDict <- tfTensorTypeDict (SingleScalarType (NumSingleType nt))
  = mkMapPrimAppF' $ TCast (SingleScalarType (NumSingleType (IntegralNumType it))) (SingleScalarType (NumSingleType nt))

mkMapPrimAppF (PrimToFloating _ _) = undefined

mkMapPrimAppF' :: TensorOp (In sh a -> Out sh b -> ()) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' a -> Arg env (Out sh b) -> OperationAcc TensorOp env ()
mkMapPrimAppF' op env exp aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF -- flatten higher-order expression
     (weakenEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   Exec -- apply method to the result
    op
    (
      ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>:
      weaken w aOut :>:
      ArgsNil
    )

mkMapPrimAppF2' :: PrimFun ((a, a) -> a) -> PrimFun ((a, a) -> a) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' (a, a) -> Arg env (Out sh (a, a)) -> OperationAcc TensorOp env ()
mkMapPrimAppF2' fun1 fun2 env exp (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2)) =
  Alet (LeftHandSideWildcard TupRunit) TupRunit
  (mkMapPrimAppF
    fun1
    env
    exp
    (ArgArray Out (ArrayR sh t1) gv gvb1)
  )
  (mkMapPrimAppF
    fun2
    env
    exp
    (ArgArray Out (ArrayR sh t2) gv gvb2)
  )
mkMapPrimAppF2' _ _ _ _ _ = error "impossible"

shapeSize :: ShapeR sh -> ExpVars env' sh -> PreOpenExp (ArrayInstr env) env' Int
shapeSize ShapeRz TupRunit                                = Const scalarTypeInt 1
shapeSize (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz))
  = PrimApp (PrimMul (IntegralNumType TypeInt))
      (Pair
        (shapeSize shr sh)
        (Evar sz))
shapeSize _ _                                             = error "impossible"

toIndex :: ShapeR sh -> ExpVars env' sh -> ExpVars env' sh -> PreOpenExp (ArrayInstr env) env' Int
toIndex ShapeRz TupRunit TupRunit = Const scalarTypeInt 0
toIndex (ShapeRsnoc shr) (TupRpair sh _) (TupRpair ix (TupRsingle i))
  = PrimApp (PrimAdd (IntegralNumType TypeInt))
      (Pair
        (toIndex shr sh ix)
        (PrimApp (PrimMul (IntegralNumType TypeInt))
          (Pair
            (shapeSize shr sh)
            (Evar i))))
toIndex _ _ _                     = error "impossible"

fromIndex :: ShapeR sh -> ExpVars env' sh -> ExpVars env' Int -> PreOpenExp (ArrayInstr env) env' sh
fromIndex ShapeRz TupRunit _                                = Nil
fromIndex (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz)) (TupRsingle i)
  | DeclareVars lhs w k <- declareVars $ TupRsingle scalarTypeInt
  = Pair
    (Let
      lhs
      (PrimApp (PrimQuot TypeInt) (Pair (Evar i) (Evar sz)))
      (fromIndex shr (weakenVars w sh) (k weakenId)))
    (case shr of
      ShapeRz -> Evar i
      _       -> PrimApp (PrimRem TypeInt) (Pair (Evar i) (Evar sz))
    )
fromIndex _ _ _              = error "impossible"

indexSlice :: SliceIndex slix t co sh1 -> ExpVars env' slix -> ExpVars env' sh1 -> PreOpenExp (ArrayInstr env) env' t
indexSlice SliceNil _ _ = Nil
indexSlice (SliceAll slix) (TupRpair slx TupRunit) (TupRpair sl (TupRsingle sz)) =
  Pair (indexSlice slix slx sl) (Evar sz)
indexSlice (SliceFixed slix) (TupRpair slx _) (TupRpair sl _) =
  indexSlice slix slx sl
indexSlice _ _ _ = error "impossible"

indexFull :: SliceIndex slix sl co t -> ExpVars env' slix -> ExpVars env' sl -> PreOpenExp (ArrayInstr env) env' t
indexFull SliceNil _ _ = Nil
indexFull (SliceAll slix) (TupRpair slx TupRunit) (TupRpair sl (TupRsingle sz)) =
  Pair (indexFull slix slx sl) (Evar sz)
indexFull (SliceFixed slix) (TupRpair slz (TupRsingle sz)) sl =
  Pair (indexFull slix slz sl) (Evar sz)
indexFull _ _ _ = error "impossible"

newtype BufferIdx benv a = BIdx (Idx benv (Buffer a))

instance Sink BufferIdx where
  weaken w (BIdx idx) = BIdx (weaken w idx)

type BufferEnv benv env = Env (BufferIdx benv) env

weakenEnv :: Sink f => (env1 :> env') -> Env (f env1) env2 -> Env (f env') env2
weakenEnv w = mapEnv (weaken w)

distributeBIdx :: forall env s. TypeR s -> GroundVars env (Buffers s) -> Distribute (BufferIdx env) s
distributeBIdx TupRunit _ = ()
distributeBIdx (TupRsingle s) (TupRsingle (Var _ idx))
  | Refl <- reprIsSingle @ScalarType @s @(BufferIdx env) s
  , Refl <- reprIsSingle @ScalarType @s @Buffer s
  = BIdx idx
distributeBIdx (TupRpair l1 l2) (TupRpair r1 r2) = (distributeBIdx l1 r1, distributeBIdx l2 r2)
distributeBIdx _ _ = error "impossible"
